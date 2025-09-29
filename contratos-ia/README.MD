\# 📄 OCR Chat com Docling + OpenAI



Este projeto de portfólio demonstra como extrair texto de arquivos PDF usando \*\*\[Docling](https://github.com/docling-ai/docling)\*\* e, em seguida, criar um \*\*chat em linha de comando (sem interface gráfica)\*\* com a \*\*API da OpenAI\*\*, permitindo conversar sobre o conteúdo do documento processado.



---



\## 🚀 Funcionalidades



\- Leitura de arquivos \*\*PDF digitalizados ou nativos\*\*.

\- Extração de texto utilizando \*\*OCR (Optical Character Recognition)\*\* via \*\*Docling\*\*.

\- Conversa interativa em \*\*modo texto\*\* (terminal) com a API da OpenAI.

\- Respostas contextuais baseadas no conteúdo do PDF processado.



---



\## 🛠️ Tecnologias Utilizadas



\- \[Python 3.10+](https://www.python.org/)

\- \[Docling](https://pypi.org/project/docling/) – OCR e parsing de documentos PDF.

\- \[OpenAI API](https://platform.openai.com/) – Geração de respostas em linguagem natural.

\- \[dotenv](https://pypi.org/project/python-dotenv/) – Gerenciamento da chave da API de forma segura.



---



\## 📂 Estrutura do Projeto



```bash

.
├── notebook/
│   └── main.py           # Script principal
├── src/
│   └── \_\_init\_\_.py
│   └── functions.py      # Contém todas as funções criadas para o projeto
│   └── common\_imports.py # Contém todos os imports
├── requirements.txt     # Dependências do projeto
├── README.md            # Documentação
├── data/
│   └── exemplo.pdf      # PDF para teste
├── data/output
│   └── exemplo.md      # PDF para teste
└── .env                 # Variáveis de ambiente (OPENAI\_API\_KEY)



