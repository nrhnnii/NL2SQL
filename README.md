# 🧠 NL2SQL Streamlit App

Turn natural language into SQL — interactively and intelligently.

This app lets you upload a SQLite database and ask questions in plain English. It retrieves the most relevant schema, picks example prompts, and uses OpenAI's GPT-4o-mini to generate accurate SQL queries — all inside a beautiful Streamlit interface.

---

## 🚀 Features

- 🗃 Upload any SQLite `.db` file
- 🔎 Uses **cosine similarity with FAISS** to find matching tables and examples
- 🤖 Powered by **OpenAI GPT-4o-mini**
- 📋 Automatically extracts and executes generated SQL queries
- 📊 Displays results in table form with preview and CSV download
- 🧠 Few-shot examples to guide SQL generation

---

## 🛠 Tech Stack

| Tool              | Purpose                           |
|-------------------|-----------------------------------|
| Streamlit         | Interactive web app UI            |
| OpenAI GPT-4o     | SQL generation via LLM            |
| FAISS             | Fast vector similarity search     |
| LangChain         | Prompting & DB access             |
| SQLite            | Database format supported         |
| Python + Pandas   | Query execution & table display   |

