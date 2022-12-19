from easyocr import Reader

# user-network


reader = Reader(['en', 'ko'], gpu=False, user_network_directory='C:/EasyOCR')



result = reader.readtext('D:/format_pc_intro_015@2x.jpg')

print(result)



