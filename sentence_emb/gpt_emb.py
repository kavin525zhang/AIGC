import openai
def get_embedding(text: str, engine="text-similarity-davinci-001"):

    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    return openai.Embedding.create(input=[text], engine=engine)["data"][0]["embedding"]

if __name__ == "__main__":
    get_embedding("cat", "text-similarity-davinci-001")