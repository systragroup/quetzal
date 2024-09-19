class MDFILE():
    def __init__(self):
        self.content=''''''
    def __repr__(self):
        return self.content
    
    def newline(self, n:int=1):
        self.content += n*'\n'
    
    def add_text(self, string:str):
        self.newline()
        self.content += string    
    
    def add_header(self, string:str, level:int=1):
        self.newline()
        formated_string = level*'#' + ' '+ string
        self.content += formated_string
        
    def add_image(self, path:str, text:str='alt text'):
        self.newline(2)
        self.content += f'![{text}]({path})'
        
    def load(self, file_path:str='sample.md'):
        with open(file_path, 'r') as mdfile:
            self.content = mdfile.read()
            
    def write(self, file_path:str='sample.md'):
        # Write the string to the markdown file
        with open(file_path, 'w') as mdfile:
            mdfile.write(self.content)