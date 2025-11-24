from MaterialAndNutritionOCR import MaterialAndNutritionImageToText

mnocr = MaterialAndNutritionImageToText()

mnocr.load_nutrition_yolo()
mnocr.load_material_yolo()
mnocr.load_easyocr()

nutrition_result1, material_result1 = mnocr.execute("material_asset/1.png")
nutrition_result2, material_result2 = mnocr.execute("nutrition_asset/1.png")

print(nutrition_result1)
print(material_result1)

print(nutrition_result2)
print(material_result2)







