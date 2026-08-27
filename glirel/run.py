from glirel import GLiREL

model = GLiREL.from_pretrained("jackboyla/glirel-large-v0")

tokens = ["Marco", "Polo", "was", "the", "co-founder", "of", "the", "Great", "Khan", "."]
labels = ["co-founder", "no relation"]
ner = [[0, 1, "PERSON", "Marco Polo"], [6, 8, "ORG", "the Great Khan"]]

relations = model.predict_relations(tokens, labels, threshold=0.0, ner=ner, top_k=1)
print(relations)
