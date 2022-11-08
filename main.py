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