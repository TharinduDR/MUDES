from mudes.app.mudes_app import MUDESApp

# app = HateSpansApp("multilingual-large", use_cuda=False)
# print(app.predict_hate_spans("ගෝඨාභය පොන්නයා විවාදෙට එන්නේ නැත්තේ ඇයි දන්නවා ද?", spans=True, language="xx"))
#
# tokens = app.predict_tokens("ගෝඨාභය පොන්නයා විවාදෙට එන්නේ නැත්තේ ඇයි දන්නවා ද?", language="xx")
# for token in tokens:
#     print(token.text, token.is_toxic)

app = MUDESApp("en-large", use_cuda=False)
print(app.predict_toxic_spans("Fuck this motherfucker", spans=True, language="en"))

tokens = app.predict_tokens("Fuck this motherfucker", language="en")
for token in tokens:
    print(token.text, token.is_toxic)