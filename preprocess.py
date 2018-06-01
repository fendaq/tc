from Utils.processor import Preprocessor

preprocessor = Preprocessor()

preprocessor.process_file_pipline()
del preprocessor.model
preprocessor.save()