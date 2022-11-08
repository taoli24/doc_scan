from transformers import LayoutLMv3FeatureExtractor
from PIL import Image, ImageDraw, ImageFont
# import matplotlib.pyplot as plt
from transformers import trainer

# Document can be a png, jpg, etc. PDFs must be converted to images.
image = Image.open("./invoice.png").convert("RGB")

# option 1: with apply_ocr=True (default)
feature_extractor = LayoutLMv3FeatureExtractor()
encoding = feature_extractor(image, do_resize=False, size={"width": image.width, "height": image.height})
print(encoding["boxes"][0])
print(encoding)


# print(encoding['pixel_values'].shape)

# plt.imshow(encoding['pixel_values'][0][1])
# plt.show()
# print(encoding.__dir__())


# dict_keys(['pixel_values', 'words', 'boxes'])

# option 2: with apply_ocr=False
# feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
# encoding = feature_extractor(image, return_tensors="pt")
# print(encoding.keys())
# # dict_keys(['pixel_values'])

def show_boxes(img, bounding_boxes):
    """
    Modify the passed in image and display bounding boxes detected by kraken on the image
    :param img: A PIL.Image object
    :param bounding_boxes: bounding boxes
    :return img: Modified PIL.Image object
    """
    from PIL import ImageDraw
    # print(image.width, image.height)
    h_scale = image.height / 1000
    w_scale = image.width / 1000

    b_boxes = bounding_boxes[0]

    for box in b_boxes:
        box[0], box[2] = box[0] * w_scale, box[2] * w_scale
        box[1], box[3] = box[1] * h_scale, box[3] * h_scale

    print(b_boxes)
    # Draw bounding boxed onto the image
    draw_object = ImageDraw.Draw(img)
    for box in b_boxes:
        draw_object.rectangle(tuple(box), fill=None, outline='red')

    return img, b_boxes


im, bounding_boxes = show_boxes(image, encoding["boxes"])
im.save("output.png")


def display_text(raw_image, bounding_boxes, text):
    img = Image.new(size=(raw_image.width, raw_image.height), mode="RGB", color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # font = ImageFont.true-type(<font-file>, <font-size>)
    font = ImageFont.truetype("Roboto-Regular.ttf", 16)
    # draw.text((x, y),"Sample Text",(r,g,b))

    for i, box in enumerate(bounding_boxes):
        draw.text((box[0], box[1]), text[i], (0, 0, 0), font=font)

    img.save("extracted_text.png")


display_text(image, bounding_boxes, encoding['words'][0])


# print(len(bounding_boxes), len(encoding['words'][0]))


def export_json(words, bounding_boxes, output_file):
    import json
    result = []
    for i, box in enumerate(bounding_boxes):
        result.append({
            "key": i,
            "text": words[i],
            "bounding_box": box
        })

    with open(output_file, mode="w") as f:
        json.dump(result, f, indent=6)


export_json(encoding['words'][0], bounding_boxes, "output.json")
