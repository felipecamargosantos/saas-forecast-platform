# SaaS Forecast Platform

Plataforma SaaS assíncrona para previsão de séries temporais desenvolvida com FastAPI, TensorFlow (LSTM) e autenticação JWT. Projetada para processar treinamentos pesados de Deep Learning em segundo plano, evitando bloqueios na API.

## Funcionalidades
- **API REST Assíncrona:** Upload de dados e enfileiramento de treinamento via `BackgroundTasks` (retorno imediato `HTTP 202 Accepted`).
- **Deep Learning (LSTM):** Redes neurais recorrentes para previsão de demanda com janelamento temporal e validação sem *data leakage*.
- **Inferência Autoregressiva:** Projeção multi-step para múltiplos dias futuros realimentando iterativamente o modelo.
- **Segurança Corporativa:** Autenticação stateless com JWT e hash de senhas utilizando `bcrypt`.
- **Qualidade e Deploy:** Testes unitários com `pytest` e empacotamento pronto para produção via Docker.

## Stack Tecnológica
- **Backend:** Python, FastAPI, Pydantic, SQLAlchemy, Uvicorn
- **Machine Learning:** TensorFlow, Keras, Scikit-Learn, Pandas, NumPy
- **Segurança & Testes:** Python-Jose, Bcrypt, Pytest, Docker

## Como Executar

```bash
# Clone o repositório
git clone [https://github.com/SEU_USUARIO/saas-forecast-platform.git](https://github.com/SEU_USUARIO/saas-forecast-platform.git)
cd saas-forecast-platform

# Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt

# Inicie o servidor da API
uvicorn app.main:app --reload --port 8000

# Execute os testes automatizados
pytest tests/ -v
