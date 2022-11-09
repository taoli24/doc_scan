import json
import os

import pypdfium2 as pdfium
from transformers import LayoutLMv3FeatureExtractor
from PIL import Image, ImageDraw, ImageFont

import utils


def testDocquery():
    from docquery import document
    from docquery.transformers_patch import pipeline

    p = pipeline('document-question-answering')
    doc = document.load_document("./invoice_easy.pdf")
    questions = [
        "What is the invoice number?",
        "What is the invoice number?",
        "What is the invoice number?",
        "What is the invoice total?",
        "What is the GST amount?",
        "What is the invoice date?",
        "What is the due date?",
        "What is the customer?",
        "Who is the supplier?",
        "What is the trading terms?",
        "What is supplier ABN?",
        "What is supplier address?",
    ]

    for q in questions:
        print(q, p(question=q, **doc.context))


def convertDataset(dataset_dir: str, save_dir: str, overwrite: bool = False, **kwargs) -> None:
    """
    Convert a dataset of pdf documents to the desired image type.

    :param dataset_dir: Path to the dataset documents, should be a str
    :param save_dir: Save path for the converted documents, should be a str
    :param overwrite: Whether to overwrite the exiting files, should be a bool
    :param kwargs: Keywords and values to be passed to convertPDF
    :return: None
    """
    for document_dir in utils.listPath(dataset_dir, ext=['pdf', 'PDF'], return_file_path=True)[1]:
        document_name = utils.getLastPath(document_dir)
        pdf2png_path, exist = utils.checkPath(save_dir, "pdf2img", os.path.splitext(document_name)[0],
                                              errors='ignore')
        if not exist or overwrite:
            print('converting document:', document_name)
            convertPDF(document_dir, pdf2png_path, **kwargs)
    print('Documents have been converted and saved!')


def convertPDF(document_dir: str, pdf2img_path: str, ext: str = 'png') -> None:
    """
    Convert the pdf file into Images, to be saved as separate pages.

    :param document_dir: Path to pdf document, should be a str
    :param pdf2img_path: Path to save the converted pdfs, should be a str
    :param ext: Save the images with file extension, should be a str
    :return: None
    """
    utils.makePath(pdf2img_path)
    pdf = pdfium.PdfDocument(document_dir)
    page_indices = [i for i in range(len(pdf))]
    renderer = pdf.render_to(pdfium.BitmapConv.pil_image, page_indices=page_indices)
    for pg_num, image in zip(page_indices, renderer):
        image.save(utils.joinPath(pdf2img_path, f"page-{pg_num}", ext=ext))


def rescaleBBoxes(img: Image, bboxes: list) -> list:
    """
    Rescales the bounding boxes.

    :param img: The document page, should be an Image
    :param bboxes: The documents bounding boxes, should be a list[list[int | float]]
    :return: bboxes - list[list[int | float]]
    """
    w_scale, h_scale = img.width / 1000, img.height / 1000
    for box in bboxes:
        box[0], box[2] = box[0] * w_scale, box[2] * w_scale
        box[1], box[3] = box[1] * h_scale, box[3] * h_scale
    return bboxes


def showBoxes(img, bboxes) -> Image:
    """
    Modify the passed in image and display bounding boxes detected by kraken on the image
    :param img: Single page image from a document, should be an Image
    :param bboxes: Bounding boxes of each extracted word from page, should
    be a list[list[x1, y1, x2, y2]]
    :return img: Modified PIL.Image object
    """
    # Draw bounding boxed onto the image
    draw_object = ImageDraw.Draw(img)
    for box in bboxes:
        draw_object.rectangle(tuple(box), fill=None, outline='red')


def displayText(raw_img_dims: tuple, bboxes: list, words: list, show_bboxes: bool = True) -> Image:
    """
    Draws the extracted words to a black canvas, then saves the image for
    visual comparisons.

    :param raw_img_dims: Dimensions of a single page image, should be a tuple
    :param bboxes: Bounding boxes of each extracted word from page, should
    be a list[list[x1, y1, x2, y2]]
    :param words: The extracted words form page, should be list[str]
    :param show_bboxes: Whether to show the bounding boxes, should be a bool
    :return: img - Image
    """
    img = Image.new(size=raw_img_dims, mode="RGB", color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    for box, word in zip(bboxes, words):
        fontsize = 1  # starting font size
        font = ImageFont.truetype("Roboto-Regular.ttf", fontsize)
        box_length = box[2] - box[0]
        while font.getlength(word) < box_length:
            fontsize += 1
            font = ImageFont.truetype("Roboto-Regular.ttf", fontsize)
        font = ImageFont.truetype("Roboto-Regular.ttf", fontsize-1)
        draw.text((box[0], box[1]), word, (0, 0, 0), font=font)
        if show_bboxes:
            draw.rectangle(tuple(box), fill=None, outline='red')
    return img


def exportJson(words, bboxes, output_file):
    result = []
    for i, box in enumerate(bboxes):
        result.append({
            "key": i,
            "text": words[i],
            "bbox": box
        })

    with open(output_file, mode="w") as f:
        json.dump(result, f, indent=4)


def testTransformers():
    # Document can be a png, jpg, etc. PDFs must be converted to images.
    image = Image.open("test/invoice.png").convert("RGB")

    feature_extractor = LayoutLMv3FeatureExtractor()
    encoding = feature_extractor(image, do_resize=False, size={"width": image.width, "height": image.height})

    page = 0
    bboxes = rescaleBBoxes(image, encoding["boxes"][page])
    showBoxes(image, bboxes)
    image.save("test/output.png")

    text_img = displayText(image.size, bboxes, encoding['words'][page], show_bboxes=False)
    text_img.save("test/extracted_text.png")

    exportJson(encoding['words'][page], bboxes, "test/output.json")


def main():
    # testDocquery()
    # testTransformers()
    dir_ = "C:\\Users\\green\\Desktop\\Xaana.ai"
    dataset_dir = utils.joinPath(dir_, 'invoice datatset\\CASA')
    results_dir = utils.makePath(dir_, 'doc_scan\\results')

    convertDataset(dataset_dir, results_dir, overwrite=False, ext='png')

    for document_dir in utils.listPath(dataset_dir, ext=['pdf', 'PDF'], return_file_path=True)[1]:
        document_name = utils.getLastPath(document_dir, include_ext=False)
        outputs_dir = utils.makePath(results_dir, 'layoutml_outputs', document_name)
        print("Processing document:", document_name)
        for page_dir in utils.listPath(results_dir, 'pdf2img', document_name, return_file_path=True)[1]:
            img_dir, exist = utils.checkPath(page_dir, ext='png')
            if exist:
                image = Image.open(img_dir).convert("RGB")

                feature_extractor = LayoutLMv3FeatureExtractor()
                encoding = feature_extractor(image, do_resize=False,
                                             size={"width": image.width, "height": image.height})

                page = 0  # scanning 1 page at a time
                bboxes = rescaleBBoxes(image, encoding["boxes"][page])
                showBoxes(image, bboxes)
                image.save(utils.joinPath(outputs_dir, "bboxes_on_raw.png"))

                text_img = displayText(image.size, bboxes, encoding['words'][page])
                text_img.save(utils.joinPath(outputs_dir, "extracted_text.png"))

                exportJson(encoding['words'][page], bboxes, utils.joinPath(outputs_dir, "extracted_text.json"))


if __name__ == "__main__":
    main()
