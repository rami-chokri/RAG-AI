from flask import Flask, render_template, request, jsonify
from rag_query import ask_rag

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Veuillez entrer une question.", "sources": []})

    answer, sources = ask_rag(question)
    return jsonify({"answer": answer, "sources": sources})


if __name__ == "__main__":
    app.run(debug=True)
