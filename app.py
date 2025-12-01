import os
import traceback
import numpy as np
import gradio as gr

from openai import AsyncOpenAI
from langsmith import traceable
from sklearn.metrics.pairwise import cosine_similarity

from src.prompts import system_prompt, evaluator_system_prompt
from src.name_extractor import extract_name_gliner
from src.models import Evaluation, CacheEntry
from src.config import Config
from src.utils import FileReader

# ---------------------------------------------------------------------
# CHAT CLASS
# ---------------------------------------------------------------------
class MyProfileAvatarChat(Config, FileReader):
    def __init__(self, max_history_turns: int = 10, similarity_thresh: float = 0.80):
        Config.__init__(self)
        FileReader.__init__(self)

        # 1. Try to load from env
        self.name = os.getenv("PROFIL_NAME")
        if not self.name:
            name = extract_name_gliner(self.linkedin_profile)
            self.name = name["person"][0]
            print(f"Name found on Linkedin profile: {self.name}")

        self.openai = AsyncOpenAI(api_key=self.openai_api_key)
        # gemini (evaluator) uses google_api_key via OpenAI wrapper
        self.gemini = AsyncOpenAI(api_key=self.google_api_key, 
                             base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        
        # Build system prompt once
        self.system_prompt = system_prompt
        self.system_prompt += f"## Linkedin Profile:\n{self.linkedin_profile}\n\n"
        self.system_prompt += f"## Addidional Information:\n{self.additional_info}\n\n"
        self.system_prompt += f"With this context, please chat with user, always staying in character as {self.name}."

        self.evaluator_system_prompt = evaluator_system_prompt

        # Settings
        self.max_history_turns = max_history_turns
        self.similarity_threshold = similarity_thresh

        # QA cache (question -> answer -> embedding)
        self.qa_cache = [] # list of dict: {"question": str, "answer": str, "embedding": np.array}
        
    
    def format_history(self, history):
        return "\n".join(f"{turn['role'].upper()}: {turn['content']}" for turn in history)
    
    async def embed(self, text: str):
        """Return embedding vector for text (uses OpenAI embeddings)."""
        resp = await self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(resp.data[0].embedding)
    
    def cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0])
    
    async def find_similar_question(self, new_question: str):
        if not self.qa_cache:
            return None, 0.0
        new_emb = await self.embed(new_question)
        best = None
        best_sim = 0.0
        for item in self.qa_cache:
            sim = self.cosine_sim(new_emb, item["embedding"])
            if sim > best_sim:
                best_sim = sim
                best = item
        if best and best_sim >= self.similarity_threshold:
            return best, best_sim
        return None, best_sim
    
    def evaluator_user_prompt(self, reply, message, history):
        formatted_history = self.format_history(history)
        user_prompt = f"Here's the conversation between the User and the Agent: \n\n{formatted_history}\n\n"
        user_prompt += f"Here's the latest message from the User: \n\n{message}\n\n"
        user_prompt += f"Here's the latest response from the Agent: \n\n{reply}\n\n"
        user_prompt += f"Please evaluate the response, replying with whether it is acceptable and your feedback."
        return user_prompt

    @traceable(run_type="tool", name="EvaluateReply")
    async def evaluate(self, reply, message, history, **kwargs) -> Evaluation:
        messages = [{"role": "system", "content": self.evaluator_system_prompt}] + \
                    [{"role": "user", "content": self.evaluator_user_prompt(reply, message, history)}]
        response = await self.gemini.chat.completions.parse(
            model="gemini-2.0-flash",
            messages=messages,
            response_format=Evaluation
        )
        return response.choices[0].message.parsed
    
    @traceable(run_type="llm", name="RerunRejectedAnswer")
    async def rerun(self, reply, message, history, feedback, **kwargs):
        updated_system_prompt = (
            self.system_prompt 
            + "\n\n## Previous answer rejected\n"
            + "You just tried to reply, but the quality control rejected your reply\n"
            + f"## Your attempted answer:\n{reply}\n\n"
            + f"## Reason for rejection:\n{feedback}\n\n"
        )
        messages = [{"role": "system", "content": updated_system_prompt}] + history + \
                    [{"role": "user", "content": message}]
        try:
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during rerun: {e}")
            return reply
        
    async def chat(self, message: str, history: list, **kwargs):
        """Main chat. Uses semantic QA cache, sliding window for tokens, evaluation and rerun
        
        Args:
            message: user message string
            history: existing list of dicts [{"role":...., "content":....}]
        Returns:
            reply string
        """
        # Cache exact-match short-circuit
        if message in (qa["question"] for qa in self.qa_cache):
            # exact match
            for qa in self.qa_cache:
                if qa["question"] == message:
                    print("Using exact cached reply")
                    history.append({"role": "user", "content": message})
                    history.append({"role": "assistant", "content": qa["answer"]})
                    return qa["answer"]
                
        # Check for semantically similar previous question
        similar, sim_score = await self.find_similar_question(message)
        if similar:
            print(f"Reusing past answer (similarity={sim_score:.2%})")
            refine_prompt = (
                f"The user previously asked a similar question:\n"
                + f"Old question: {similar['question']}\n"
                + f"Old answer: {similar['answer']}\n\n"
                + f"Now user asks: {message}\n\n"
                + f"Please update or refine the old answer to match the new question."
            )
            messages = [{"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": refine_prompt}]
            try:
                response = await self.openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages
                )
                reply = response.choices[0].message.content
            except Exception as e:
                print(f"Error calling OpenAI for refinement: {e}")
                reply = similar["answer"]  
        else:
            # Build token-efficent context (sliding window)
            temp_history = history + [{"role": "user", "content": message}]
            context_for_api = temp_history[-self.max_history_turns:]
            messages = [{"role": "system", "content": self.system_prompt}] + context_for_api

            try:
                response = await self.openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages
                )
                reply = response.choices[0].message.content
            except Exception as e:
                print(f"Error calling OpenAI: {e}")
        # Evaluate the reply
        try:
            evaluation = await self.evaluate(reply, message, history)
        except Exception as e:
            print(f"Error during evaluation: {e}")
            evaluation = Evaluation(is_acceptable=True, feedback="Evaluation failed, accepting reply")

        if evaluation:
            print(f"Feedback from Evaluation:\n{evaluation.feedback}\n\n")
        if not evaluation.is_acceptable:
            reply = await self.rerun(reply, message, history, evaluation.feedback)

        try:
            emb = await self.embed(message)
        except Exception as e:
            print(f"Embedding Error: {e}")
            traceback.print_exc()
            emb = None
        
        self.qa_cache.append(CacheEntry(
            question=message,
            answer=reply,
            embedding=emb.tolist() if hasattr(emb, "tolist") else emb
        ))

        return reply
    
    @traceable(run_type="chain", name="ProfileChat")
    async def chat_traced(self, *args, **kwargs):
        """Wrapper for LangSmith tracing. Accepts any extra arguments
        (like from Gradio) and passes only message/history to chat()."""

        if len(args) >=2:
            message, history = args[0], args[1]
        else:
            message = kwargs.get("message")
            history = kwargs.get("history")
        return await self.chat(message, history)
        
if __name__ == "__main__":

    print("HF rebuild test – App.py loaded successfully.")

    my_profile = MyProfileAvatarChat()
    gr.ChatInterface(my_profile.chat_traced,type="messages").launch()

