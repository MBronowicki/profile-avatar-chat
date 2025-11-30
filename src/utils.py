from src.file_loader import load_pdf_text, load_text_file

# ---------------------------------------------------------------------
# FILE READER
# ---------------------------------------------------------------------
class FileReader:
    def __init__(self):
        self.linkedin_profile = load_pdf_text("./me/Linkedin_Profile.pdf")
        self.additional_info = load_text_file("./me/additional_info.txt")
    
        # print("=== LINKEDIN PROFILE CONTENT ===")
        # print(self.linkedin_profile)
        # print("=== END ===")