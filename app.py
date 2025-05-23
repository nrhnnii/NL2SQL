import os
import re
import time
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate

load_dotenv()

examples_pool = [
    {"input": "Show all records from a table.", "query": "SELECT * FROM Table;"},
    {"input": "Get the names and prices of items that cost more than 100.", "query": "SELECT Name, Price FROM Items WHERE Price > 100;"},
    {"input": "Count the number of entries in a table.", "query": "SELECT COUNT(*) FROM Table;"},
    {"input": "Find the total sales amount.", "query": "SELECT SUM(Amount) AS TotalSales FROM Sales;"},
    {"input": "How many items belong to each category?", "query": "SELECT Category, COUNT(*) AS ItemCount FROM Items GROUP BY Category;"},
    {"input": "Which customers have made more than 3 purchases?", "query": "SELECT CustomerID, COUNT(*) AS PurchaseCount FROM Orders GROUP BY CustomerID HAVING COUNT(*) > 3;"},
    {"input": "What is the average price of products?", "query": "SELECT AVG(Price) AS AveragePrice FROM Products;"},
    {"input": "List the top 5 products by total sales.", "query": "SELECT ProductID, SUM(Sales) AS TotalSales FROM Transactions GROUP BY ProductID ORDER BY TotalSales DESC LIMIT 5;"},
    {"input": "Get unique product names.", "query": "SELECT DISTINCT Name FROM Products;"},
    {"input": "Show the first order by date.", "query": "SELECT * FROM Orders ORDER BY OrderDate LIMIT 1;"},
    {"input": "Show all employees with the title 'Sales Representative'.", "query": "SELECT Name, Title FROM Employees WHERE Title = 'Sales Representative';"},
    {"input": "Show all products", "query": "SELECT * FROM Product;"}
]

encoder = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

def build_faiss_index(vectors):
    dim = len(vectors[0])
    vectors = np.array([v / np.linalg.norm(v) for v in vectors]).astype('float32')
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index, vectors

def get_examples_faiss(query, k=3):
    example_texts = [ex["input"] for ex in examples_pool]
    example_vectors = [encoder.embed_query(text) for text in example_texts]
    index, _ = build_faiss_index(example_vectors)
    query_vec = encoder.embed_query(query)
    query_vec = np.array(query_vec) / np.linalg.norm(query_vec)
    query_vec = query_vec.astype("float32").reshape(1, -1)
    distances, indices = index.search(query_vec, k)
    return [examples_pool[i] for i in indices[0]]

def get_schema_faiss(query, db, k=4):
    table_info = db.get_table_info()
    table_blocks = table_info.split("\n\n")
    block_vectors = [encoder.embed_query(block) for block in table_blocks]
    index, _ = build_faiss_index(block_vectors)
    query_vec = encoder.embed_query(query)
    query_vec = np.array(query_vec) / np.linalg.norm(query_vec)
    query_vec = query_vec.astype("float32").reshape(1, -1)
    distances, indices = index.search(query_vec, k)
    return "\n\n".join([table_blocks[i] for i in indices[0]])

def extract_sql(text):
    match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = text.strip().splitlines()
    sql_lines = [line for line in lines if line.strip().upper().startswith(("SELECT", "WITH", "INSERT", "UPDATE", "DELETE"))]
    return " ".join(sql_lines).strip() if sql_lines else text.strip()

def main():
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center;'>NL2SQL</h1>", unsafe_allow_html=True)

    file = st.sidebar.file_uploader("📂 Upload SQLite Database", type="db")
    db = None

    if file:
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        temp_path.write(file.read())
        temp_path.close()
        db = SQLDatabase.from_uri(f"sqlite:///{temp_path.name}")
        st.sidebar.success("✅ Connected to SQLite!")

    if db:
        st.markdown("### 💬 Ask Your Database")
        query = st.text_area("Query Input", label_visibility="collapsed", height=68)

        if st.button("Submit") and query.strip():
            try:
                schema = get_schema_faiss(query, db)
                examples = get_examples_faiss(query)
                formatted_examples = "\n".join([f"User input: {ex['input']}\nSQL query: {ex['query']}" for ex in examples])

                prompt_text = f"""You are a SQL expert. Create a correct query using the top {{top_k}} most relevant tables.

Relevant schema:
{{table_info}}

Examples:
{formatted_examples}

User input: {{input}}
SQL query:"""

                dyn_prompt = PromptTemplate(
                    input_variables=["input", "top_k", "table_info"],
                    template=prompt_text
                )

                chain = create_sql_query_chain(llm=llm, db=db, prompt=dyn_prompt)
                response = chain.invoke({
                    "question": query,
                    "input": query,
                    "table_info": schema,
                    "top_k": 5
                })

                raw_sql = extract_sql(response)
                raw_conn = db._engine.raw_connection()
                cursor = raw_conn.cursor()
                try:
                    cursor.execute(raw_sql)
                    all_rows = cursor.fetchall()
                    column_names = [desc[0] for desc in cursor.description]
                    cursor.close()
                    raw_conn.close()

                    st.success("✅ SQL query executed successfully.")
                    st.markdown("### 📄 SQL Query Generated")
                    st.code(raw_sql, language="sql")

                    df = pd.DataFrame(all_rows, columns=column_names)
                    st.markdown("### 📊 Query Result Preview")
                    st.dataframe(df)

                    n_rows, n_cols = df.shape
                    row_label = "row" if n_rows == 1 else "rows"
                    col_label = "column" if n_cols == 1 else "columns"
                    st.markdown(f"**{n_rows} {row_label} × {n_cols} {col_label}**")

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Full Results as CSV", csv, "query_results.csv", "text/csv")
                except Exception as e:
                    st.error(f"❌ SQL execution failed: {e}")
                    cursor.close()
                    raw_conn.close()

            except Exception as e:
                st.error(f"⚠️ Something went wrong: {e}")

if __name__ == "__main__":
    main()
