
from common_imports import *

#Executa OCR de arquivos
def executa_ocr(input_doc_path):
    """Realiza a leitura de arquivos pdf que contenham texto, imagens... e armazena um arquivo .md (markdown) com o texto extraído"""
    # Configurações do pipeline
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    # Configurações do OCR
    ocr_options = TesseractCliOcrOptions(force_full_page_ocr=False)
    ocr_options.tesseract_cmd = r"INFORME O CAMINHO\tesseract.exe"
    pipeline_options.ocr_options = ocr_options

    # Inicializa o conversor
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    output_dir = Path("../data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = converter.convert(input_doc_path)
        doc = result.document
        md = doc.export_to_markdown()
        output_file = output_dir / (input_doc_path.stem + ".md")
        output_file.write_text(md, encoding="utf-8")
        print(f"Markdown salvo: {output_file}")
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {input_doc_path}")
    except Exception as e:
        print(f"Erro ao processar {input_doc_path}: {e}")

# Realiza a leitura do arquivo markdown
def ler_arquivo_md(caminho):
    with open(caminho, "r", encoding="utf-8") as f:
        conteudo = f.read()
    print(str(conteudo)[:700])
    return conteudo

# Faz o split do texto separando por markdonw
def gera_split_texto(conteudo):
    headers_to_split_on = [
        ("#", "Header"),
        ("##", "Header1"),
        ("###", "Header2"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on,
        return_each_line=True,
    )
    docs = markdown_splitter.split_text(conteudo)
    return docs

# Realiza o login no HuggingFace
def login_huggingface():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)

# Inicializa o modelo da OpenAI com a API KEY e gera alguns resultados
def responder_gpt(chat, contex):
    login_huggingface()
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    client = OpenAI()
    prompt = f"""
    Você é um assistente jurídico. Use apenas o contexto abaixo para responder, sem inventar informações:

    {contex}

    Pergunta: {chat}
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content