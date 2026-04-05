
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Google Gemini APIのインポート（APIキーが設定されている場合のみ使用）
try:
    import google.generativeai as genai
except ImportError:
    genai = None
    st.warning("Google Generative AI SDKがインストールされていません。LLMモードは利用できません。`pip install google-generativeai` でインストールしてください。")

# --- データロード関数 ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"データファイルの読み込み中にエラーが発生しました: {e}")
        return pd.DataFrame()

# --- RAG（検索拡張生成）ロジック --- 
@st.cache_resource
def setup_rag(df):
    if df.empty:
        return None, None
    
    # 検索対象となるテキストを結合
    df['search_text'] = df['主な対応キーワード'].fillna('') + ' ' + df['業務の具体的な内容'].fillna('')
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['search_text'])
    return vectorizer, tfidf_matrix

def search_relevant_info(query, df, vectorizer, tfidf_matrix, top_n=3):
    if vectorizer is None or tfidf_matrix is None:
        return pd.DataFrame()

    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # 類似度が高い順にソートし、インデックスを取得
    related_docs_indices = cosine_similarities.argsort()[:-top_n-1:-1]
    
    # 類似度スコアが0より大きいもののみをフィルタリング
    relevant_indices = [idx for idx in related_docs_indices if cosine_similarities[idx] > 0]
    
    return df.iloc[relevant_indices]

# --- Streamlit UI --- 
st.set_page_config(page_title="相談ナビゲーションAI (実証実験)", layout="wide")
st.title("自治体向け 相談ナビゲーションAI (実証実験)")
st.markdown("窓口・電話相談において、適切な担当部署やマニュアルを即座に提示します。")

# サイドバー
st.sidebar.header("設定")
api_key = st.sidebar.text_input("Google Gemini APIキー", type="password", help="LLMモードで使用します。未入力の場合はモックモードで動作します。")

# APIキーが入力されているか確認
llm_mode_enabled = False
if api_key and genai:
    try:
        genai.configure(api_key=api_key)
        llm_mode_enabled = True
        st.sidebar.success("LLMモードが有効です。")
    except Exception as e:
        st.sidebar.error(f"APIキーの構成に失敗しました: {e}")
        llm_mode_enabled = False
elif not api_key:
    st.sidebar.info("APIキーが未入力のため、モックモードで動作します。")

# データソースの選択
data_source_option = st.sidebar.radio(
    "データソースを選択",
    ("アプリ内蔵ダミーデータ", "CSV/Excelファイルをアップロード"),
    index=0
)

df = pd.DataFrame()
if data_source_option == "アプリ内蔵ダミーデータ":
    df = load_data("consultation_data.csv")
else:
    uploaded_file = st.sidebar.file_uploader("CSVまたはExcelファイルをアップロード", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.sidebar.success("ファイルを正常に読み込みました。")
        except Exception as e:
            st.sidebar.error(f"ファイルの読み込み中にエラーが発生しました: {e}")

vectorizer, tfidf_matrix = setup_rag(df)

# チャット履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# チャット履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザー入力の処理
if prompt := st.chat_input("相談内容を入力してください..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if df.empty:
            st.warning("データが読み込まれていません。サイドバーからデータソースを設定してください。")
        else:
            relevant_info = search_relevant_info(prompt, df, vectorizer, tfidf_matrix)
            
            if relevant_info.empty:
                response = "申し訳ありません、関連する情報を見つけることができませんでした。別のキーワードでお試しください。"
            else:
                if llm_mode_enabled:
                    # LLMモード
                    try:
                        model = genai.GenerativeModel('gemini-pro') # または 'gemini-1.5-flash' など
                        # プロンプトエンジニアリング
                        llm_prompt = f"""
                        以下の相談内容に対して、提供された情報に基づいて、最も適切と思われる部署と対応内容を自然な言葉で案内してください。
                        もし情報が複数ある場合は、優先順位をつけて提示し、職員が次に取るべき行動を明確にしてください。
                        参照URLがあれば、それも提示してください。

                        相談内容: {prompt}

                        関連情報:
                        """
                        for i, row in relevant_info.iterrows():
                            llm_prompt += f"""
                            - 部署: {row['部名']} {row['課名']} {row['係名']}
                            - 主な対応キーワード: {row['主な対応キーワード']}
                            - 業務内容: {row['業務の具体的な内容']}
                            - 内線番号: {row['内線番号']}
                            - 参照URL: {row['参照URL']}
                            """
                        llm_prompt += f"""

                        上記の情報を踏まえて、相談者への案内文を作成してください。
                        """
                        
                        llm_response = model.generate_content(llm_prompt)
                        response = llm_response.text
                    except Exception as e:
                        response = f"LLMモードでエラーが発生しました。モックモードで回答します。エラー: {e}\n\n"
                        # エラー時はモックモードにフォールバック
                        response += "以下の情報が見つかりました：\n"
                        for i, row in relevant_info.iterrows():
                            response += f"- **部署**: {row['部名']} {row['課名']} {row['係名']}\n"
                            response += f"  - **業務内容**: {row['業務の具体的な内容']}\n"
                            response += f"  - **内線番号**: {row['内線番号']}\n"
                            if pd.notna(row['参照URL']): # NaNでない場合のみ表示
                                response += f"  - **参照URL**: {row['参照URL']}\n"
                        response += "\n詳細については、上記部署にお問い合わせください。"
                else:
                    # モックモード
                    response = "以下の情報が見つかりました：\n"
                    for i, row in relevant_info.iterrows():
                        response += f"- **部署**: {row['部名']} {row['課名']} {row['係名']}\n"
                        response += f"  - **業務内容**: {row['業務の具体的な内容']}\n"
                        response += f"  - **内線番号**: {row['内線番号']}\n"
                        if pd.notna(row['参照URL']): # NaNでない場合のみ表示
                            response += f"  - **参照URL**: {row['参照URL']}\n"
                    response += "\n詳細については、上記部署にお問い合わせください。"
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

