import streamlit as st
import pandas as pd
import zipfile
import tempfile
import requests
import os
import re
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from datetime import datetime
import locale

# Configuração de locale para formatação monetária
try:
    locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
except:
    locale.setlocale(locale.LC_ALL, 'Portuguese_Brazil.1252')

# Configuração da página
st.set_page_config(
    page_title="📊 Analisador de Notas Fiscais",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mapeamento de UF para nomes completos
UF_MAP = {
    'AC': 'Acre', 
    'AL': 'Alagoas', 
    'AP': 'Amapá', 
    'AM': 'Amazonas',
    'BA': 'Bahia', 
    'CE': 'Ceará', 
    'DF': 'Distrito Federal', 
    'ES': 'Espírito Santo', 
    'GO': 'Goiás', 
    'MA': 'Maranhão',
    'MT': 'Mato Grosso', 
    'MS': 'Mato Grosso do Sul', 
    'MG': 'Minas Gerais',
    'PA': 'Pará', 
    'PB': 'Paraíba', 
    'PR': 'Paraná', 
    'PE': 'Pernambuco',
    'PI': 'Piauí', 
    'RJ': 'Rio de Janeiro', 
    'RN': 'Rio Grande do Norte',
    'RS': 'Rio Grande do Sul', 
    'RO': 'Rondônia', 
    'RR': 'Roraima',
    'SC': 'Santa Catarina', 
    'SP': 'São Paulo', 
    'SE': 'Sergipe',
    'TO': 'Tocantins'
}

# Modelo Pydantic para validação
class FileInfo(BaseModel):
    filename: str
    columns: List[str]
    sample_data: str
    file_type: str  # 'resumo' ou 'detalhes'

# Configuração do LLM com timeout
def setup_llm(default_local_llm: str ="lmstudio"):
    def fetch_model(base_url) -> str:
        url = f"{base_url}/models"
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            
            model_id = data['data'][0]['id']
            match = re.search(r"/([^/]+)\.gguf$", model_id)
            if match:
                model = match.group(1)
                return model
            else:
                return '404'
        except requests.exceptions.RequestException as e:
            print(f"\nError fetching models: {e}\n=> Certifique-se de que o LM Studio está rodando na porta '1234' -> http://localhost:1234/v1\n")
            return '404'
    
    def ollama_run(base_url) -> str:
        url = base_url
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            for model_get in data.get('models', []):
                if 'mistral' in model_get.get("name"):
                    break
                else: 
                    print('O modelo mistral não encontrado. Certifique-se que o modelo está baixado.')
        except requests.exceptions.RequestException as e:
                    print(f"\nError fetching models: {e}\n=> Certifique-se de que o Mistral (Ollama) está rodando em -> https://ollama.com/library/mistral")
    
    def instantiate_lm_studio(base_url: str, model: str, temperature: float) -> ChatOpenAI:
        return ChatOpenAI(
                model=model,
                base_url=base_url,
                temperature=temperature,
                timeout=300  # 5 minutos de timeout
            )

    def instantiate_ollama(model: str, temperature: float) -> Ollama:
        return Ollama(
                model=model,
                temperature=temperature,
                timeout=300  # 5 minutos de timeout
            )

    try:
        if default_local_llm == "lmstudio":
            base_url = "http://localhost:1234/v1"
            model = fetch_model(base_url)
            if model != '404':
                llm = instantiate_lm_studio(base_url, model, temperature=0.2)
            else:
                st.error("Modelo não encontrado no LM Studio.\n\n- Certifique-se de que um modelo está rodando e disponível no LM Studio.\n- Certifique-se de que o servidor do LM Studio está rodando na porta '1234' -> http://localhost:1234/v1")
                st.write("Tentando com Ollama mistral...")
                llm = instantiate_ollama("mistral", temperature=0.2)
        else:
            try:
                # Usando Ollama diretamente
                base_url = 'http://localhost:11434/api/tags'
                base_url_ollama = ollama_run(base_url)
                if base_url_ollama != '404':
                    llm = instantiate_ollama("mistral", temperature=0.2)
                else: 
                    st.error('Modelo não encontrado. Certifique-se que o modelo está baixado.')
                    llm = instantiate_lm_studio(base_url, model, temperature=0.2)
            except TimeoutError:
                st.error('OPS! Parece que demorou demais a resposta... Tente novamente.')
            except Exception as e:
                st.error(f'Erro ao tentar se conectar com o Mistral (Ollama): ERROR -> {e}.')
        return llm
    except Exception as e:
        st.error(f"Erro ao conectar ao modelo: {e}")
        st.info("Certifique-se que o modelo está rodando localmente.")
        return None

def detect_file_type(df: pd.DataFrame) -> str:
    """Detecta se o arquivo é de resumo ou detalhes baseado nas colunas"""
    if 'VALOR NOTA FISCAL' in df.columns and 'UF DESTINATÁRIO' in df.columns:
        return 'resumo'
    elif 'VALOR UNITÁRIO' in df.columns and 'DESCRIÇÃO DO PRODUTO/SERVIÇO' in df.columns:
        return 'detalhes'
    return 'desconhecido'

def clean_numeric(value) -> float:
    """Converte valores numéricos que podem estar como strings para float"""
    if pd.isna(value) or value == '':
        return 0.0
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        # Remove todos os caracteres não numéricos exceto vírgula e ponto
        value = re.sub(r'[^\d,.-]', '', value).strip()
        
        # Verifica se está no formato brasileiro (1.234,56)
        if re.match(r'^\d{1,3}(?:\.\d{3})*,\d{2}$', value):
            return float(value.replace('.', '').replace(',', '.'))
        
        # Verifica se está no formato americano (1,234.56)
        elif re.match(r'^\d{1,3}(?:,\d{3})*\.\d{2}$', value):
            return float(value.replace(',', ''))
        
        # Verifica se tem vírgula como separador decimal
        elif ',' in value and value.count(',') == 1:
            parts = value.split(',')
            if len(parts) == 2 and parts[1].isdigit():
                return float(value.replace(',', '.'))
        
        # Tenta converter diretamente
        try:
            return float(value)
        except ValueError:
            return 0.0
    
    return 0.0

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza os nomes das colunas e tipos de dados"""
    # Normaliza nomes de colunas
    df.columns = [col.upper().strip() for col in df.columns]
    
    # Converte colunas de data
    date_cols = [col for col in df.columns if 'DATA' in col or 'EMISSÃO' in col]
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            pass
    
    return df

def process_data(files_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Processa os dados para análises específicas"""
    processed = {}
    
    # Identificar qual arquivo é qual
    for filename, df in files_data.items():
        df = normalize_dataframe(df)
        file_type = detect_file_type(df)
        
        if file_type == 'resumo':
            # Limpar e converter valores numéricos
            if 'VALOR NOTA FISCAL' in df.columns:
                df['VALOR NOTA FISCAL'] = df['VALOR NOTA FISCAL'].apply(clean_numeric)
            
            # Converter UF para nome completo
            if 'UF DESTINATÁRIO' in df.columns:
                df['ESTADO_DESTINO'] = df['UF DESTINATÁRIO'].map(UF_MAP)
            
            # Extrair data (removendo o horário)
            if 'DATA EMISSÃO' in df.columns:
                df['DATA'] = pd.to_datetime(df['DATA EMISSÃO']).dt.date
            
            # Extrair cidade do município emitente
            if 'MUNICÍPIO EMITENTE' in df.columns:
                df['CIDADE_DESTINO'] = df['MUNICÍPIO EMITENTE'].str.split('-').str[0].str.strip()
            else:
                df['CIDADE_DESTINO'] = 'Desconhecida'
            
            processed['resumo'] = df
            
        elif file_type == 'detalhes':
            # Limpar e converter valores numéricos
            for col in ['VALOR UNITÁRIO', 'VALOR TOTAL', 'QUANTIDADE']:
                if col in df.columns:
                    df[col] = df[col].apply(clean_numeric)
            
            processed['detalhes'] = df
    
    return processed

def calculate_metrics(processed_data: Dict[str, pd.DataFrame]) -> Dict[str, any]:
    """Calcula todas as métricas importantes uma vez para otimização"""
    metrics = {}
    
    if 'resumo' not in processed_data or 'detalhes' not in processed_data:
        return metrics
    
    df_resumo = processed_data['resumo']
    df_detalhes = processed_data['detalhes']
    
    # Métricas de vendas por estado
    if 'ESTADO_DESTINO' in df_resumo.columns and 'VALOR NOTA FISCAL' in df_resumo.columns:
        vendas_por_estado = df_resumo.groupby('ESTADO_DESTINO')['VALOR NOTA FISCAL'].agg(['sum', 'count']).sort_values('sum', ascending=False)
        metrics['estado_mais_vendeu'] = vendas_por_estado.index[0] if not vendas_por_estado.empty else 'Nenhum'
        metrics['total_por_estado'] = vendas_por_estado['sum'].to_dict()
        metrics['contagem_por_estado'] = vendas_por_estado['count'].to_dict()
    
    # Métricas de vendas por cidade
    if 'CIDADE_DESTINO' in df_resumo.columns and 'VALOR NOTA FISCAL' in df_resumo.columns:
        vendas_por_cidade = df_resumo.groupby('CIDADE_DESTINO')['VALOR NOTA FISCAL'].agg(['sum', 'count']).sort_values('sum', ascending=False)
        metrics['cidade_mais_vendeu'] = vendas_por_cidade.index[0] if not vendas_por_cidade.empty else 'Nenhuma'
        metrics['total_por_cidade'] = vendas_por_cidade['sum'].to_dict()
        metrics['contagem_por_cidade'] = vendas_por_cidade['count'].to_dict()
    
    # Métricas de datas
    if 'DATA' in df_resumo.columns and 'VALOR NOTA FISCAL' in df_resumo.columns:
        vendas_por_data = df_resumo.groupby('DATA')['VALOR NOTA FISCAL'].agg(['sum', 'count']).sort_values('sum', ascending=False)
        metrics['data_mais_vendas'] = vendas_por_data.index[0] if not vendas_por_data.empty else 'Nenhuma'
        metrics['total_por_data'] = vendas_por_data['sum'].to_dict()
        metrics['contagem_por_data'] = vendas_por_data['count'].to_dict()
    
    # Métricas de produtos
    if 'DESCRIÇÃO DO PRODUTO/SERVIÇO' in df_detalhes.columns:
        if 'QUANTIDADE' in df_detalhes.columns and 'VALOR TOTAL' in df_detalhes.columns:
            produtos_mais_vendidos = df_detalhes.groupby('DESCRIÇÃO DO PRODUTO/SERVIÇO').agg({
                'QUANTIDADE': 'sum',
                'VALOR TOTAL': 'sum'
            }).sort_values('QUANTIDADE', ascending=False)
            metrics['produto_mais_vendido'] = produtos_mais_vendidos.index[0] if not produtos_mais_vendidos.empty else 'Nenhum'
            metrics['quantidade_produto'] = produtos_mais_vendidos['QUANTIDADE'].to_dict()
            metrics['faturamento_produto'] = produtos_mais_vendidos['VALOR TOTAL'].to_dict()
        
        if 'VALOR UNITÁRIO' in df_detalhes.columns:
            produtos_maior_valor = df_detalhes.groupby('DESCRIÇÃO DO PRODUTO/SERVIÇO')['VALOR UNITÁRIO'].max().sort_values(ascending=False)
            metrics['produto_mais_caro'] = produtos_maior_valor.index[0] if not produtos_maior_valor.empty else 'Nenhum'
            metrics['valor_produto'] = produtos_maior_valor.to_dict()
    
    # Total geral
    if 'VALOR NOTA FISCAL' in df_resumo.columns:
        metrics['total_vendas'] = df_resumo['VALOR NOTA FISCAL'].sum()
        metrics['total_notas'] = len(df_resumo)
    
    return metrics

def format_currency(value: float) -> str:
    """Formata valores monetários corretamente"""
    try:
        return locale.currency(value, grouping=True, symbol=True)
    except:
        return f"R$ {value:,.2f}".replace('.', '|').replace(',', '.').replace('|', ',')

def analyze_data(question: str, metrics: Dict[str, any], processed_data: Dict[str, pd.DataFrame]) -> Tuple[Optional[str], Optional[str]]:
    """Realiza análises específicas baseadas nas métricas pré-calculadas"""
    question_lower = question.lower()
    df_resumo = processed_data.get('resumo', pd.DataFrame())
    
    try:
        # Análise de estados específicos
        if 'quanto' in question_lower and ('vendeu' in question_lower or 'faturou' in question_lower):
            for estado in UF_MAP.values():
                if estado.lower() in question_lower:
                    if estado in metrics.get('total_por_estado', {}):
                        total = format_currency(metrics['total_por_estado'][estado])
                        count = metrics.get('contagem_por_estado', {}).get(estado, 0)
                        return f"O estado de {estado} teve {count} notas fiscais totalizando {total}.", "estado"
                    else:
                        # Verifica diretamente no dataframe se não encontrou nas métricas
                        if 'ESTADO_DESTINO' in df_resumo.columns and 'VALOR NOTA FISCAL' in df_resumo.columns:
                            estado_df = df_resumo[df_resumo['ESTADO_DESTINO'] == estado]
                            if not estado_df.empty:
                                total = format_currency(estado_df['VALOR NOTA FISCAL'].sum())
                                count = len(estado_df)
                                return f"O estado de {estado} teve {count} notas fiscais totalizando {total}.", "estado"
                        return f"Não foram encontradas vendas para o estado de {estado}.", "estado"
        
        # Análise de estados
        if any(word in question_lower for word in ['estado', 'uf', 'local']) and 'mais' in question_lower:
            if 'estado_mais_vendeu' in metrics:
                estado = metrics['estado_mais_vendeu']
                total = format_currency(metrics['total_por_estado'][estado])
                count = metrics.get('contagem_por_estado', {}).get(estado, 0)
                return f"O estado com maior volume de vendas foi {estado} com {count} notas fiscais totalizando {total}.", "estado"
        
        # Análise de cidades
        elif any(word in question_lower for word in ['cidade', 'município']):
            if 'cidade_mais_vendeu' in metrics:
                cidade = metrics['cidade_mais_vendeu']
                total = format_currency(metrics['total_por_cidade'][cidade])
                count = metrics.get('contagem_por_cidade', {}).get(cidade, 0)
                return f"A cidade com maior volume de vendas foi {cidade} com {count} notas fiscais totalizando {total}.", "cidade"
        
        # Análise de produtos
        elif any(word in question_lower for word in ['produto', 'item', 'mercadoria']):
            if 'caro' in question_lower and 'produto_mais_caro' in metrics:
                produto = metrics['produto_mais_caro']
                valor = format_currency(metrics['valor_produto'][produto])
                return f"O produto mais caro vendido foi '{produto}' com valor unitário de {valor}.", "produto"
            elif 'produto_mais_vendido' in metrics:
                produto = metrics['produto_mais_vendido']
                quantidade = int(metrics['quantidade_produto'][produto])
                faturamento = format_currency(metrics.get('faturamento_produto', {}).get(produto, 0))
                return f"O produto mais vendido foi '{produto}' com {quantidade} unidades, gerando {faturamento}.", "produto"
        
        # Análise de datas
        elif any(word in question_lower for word in ['data', 'dia', 'mês']):
            if 'data_mais_vendas' in metrics:
                data = metrics['data_mais_vendas'].strftime('%d/%m/%Y')
                total = format_currency(metrics['total_por_data'][metrics['data_mais_vendas']])
                count = metrics.get('contagem_por_data', {}).get(metrics['data_mais_vendas'], 0)
                return f"O dia com maior volume de vendas foi {data} com {count} notas fiscais totalizando {total}.", "data"
        
        # Análise geral
        elif 'total' in question_lower and 'total_vendas' in metrics:
            total = format_currency(metrics['total_vendas'])
            count = metrics.get('total_notas', 0)
            return f"O valor total de vendas foi {total} em {count} notas fiscais.", "total"
        
        return None, None  # Retorna None para indicar que a análise automática não foi possível
    
    except Exception as e:
        return f"Erro ao analisar os dados: {str(e)}", "erro"

# Processamento do arquivo ZIP com múltiplas tentativas de leitura
def process_zip(uploaded_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            csv_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.csv')])
            
            if len(csv_files) != 2:
                st.error("O arquivo ZIP deve conter exatamente dois arquivos CSV.")
                return None, None
            
            files_data = {}
            files_info = []
            
            for csv_file in csv_files:
                file_path = os.path.join(temp_dir, csv_file)
                try:
                    # Tentar ler com diferentes combinações de encoding e separadores
                    encodings = ['utf-8', 'latin1', 'ISO-8859-1']
                    separators = [',', ';', '\t']
                    
                    for encoding in encodings:
                        for sep in separators:
                            try:
                                df = pd.read_csv(file_path, encoding=encoding, sep=sep, decimal='.', thousands=',')
                                # Verifica se leu corretamente (pelo menos algumas colunas)
                                if len(df.columns) > 1:
                                    break
                            except:
                                continue
                        else:
                            continue
                        break
                    else:
                        # Última tentativa sem especificar separador
                        try:
                            df = pd.read_csv(file_path, encoding='utf-8')
                        except:
                            df = pd.read_csv(file_path, encoding='latin1')
                    
                    file_type = detect_file_type(df)
                    
                    files_data[csv_file] = df
                    
                    file_info = FileInfo(
                        filename=csv_file,
                        columns=list(df.columns),
                        sample_data=df.head(3).to_string(),
                        file_type=file_type
                    )
                    files_info.append(file_info)
                except Exception as e:
                    st.error(f"Erro ao ler o arquivo {csv_file}: {e}")
                    return None, None
            
            return files_data, files_info
            
        except Exception as e:
            st.error(f"Erro ao processar arquivo ZIP: {e}")
            return None, None

# Interface Streamlit
def main():
    st.title("📊 Analisador Avançado de Notas Fiscais")
    
    # Inicialização de sessão
    if 'files_data' not in st.session_state:
        st.session_state.files_data = None
    if 'files_info' not in st.session_state:
        st.session_state.files_info = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory()
    
    # Configuração do LLM
    software_local = st.radio(
        "Software Local:",
        ("LM Studio", "Ollama - Mistral")
    )

    # Upload do arquivo
    uploaded_file = st.file_uploader("Envie seu arquivo ZIP com dois CSVs", type="zip")
    
    if uploaded_file and st.session_state.files_data is None:
        with st.spinner("Processando arquivo..."):
            files_data, files_info = process_zip(uploaded_file)
            if files_data and files_info:
                st.session_state.files_data = files_data
                st.session_state.files_info = files_info
                st.session_state.processed_data = process_data(files_data)
                st.session_state.metrics = calculate_metrics(st.session_state.processed_data)
                st.success("Arquivos processados com sucesso!")
                
                # Mostrar preview dos arquivos
                st.subheader("Pré-visualização dos Arquivos")
                tab1, tab2 = st.tabs([
                    f"Arquivo 1: {files_info[0].filename} ({files_info[0].file_type})", 
                    f"Arquivo 2: {files_info[1].filename} ({files_info[1].file_type})"
                ])
                
                with tab1:
                    st.dataframe(files_data[files_info[0].filename].head())
                
                with tab2:
                    st.dataframe(files_data[files_info[1].filename].head())
    
    # Chat de análise
    if st.session_state.metrics:
        st.divider()
        st.subheader("Faça suas perguntas sobre os dados")
        
        # Mostrar histórico de chat
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Input do usuário
        user_question = st.chat_input("Digite sua pergunta (ex: 'Qual estado vendeu mais?')")
        
        if user_question:
            # Adicionar pergunta ao histórico
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            with st.chat_message("user"):
                st.markdown(user_question)
            
            # Processar pergunta
            with st.spinner("Analisando dados..."):
                try:
                    # Primeiro tentamos análise automática
                    auto_response, response_type = analyze_data(
                        user_question, 
                        st.session_state.metrics,
                        st.session_state.processed_data
                    )
                    
                    if auto_response is not None and not auto_response.startswith("Erro"):
                        response = auto_response
                    else:
                        # Se a análise automática não funcionar, usamos o LLM
                        if software_local == "LM Studio":
                            llm = setup_llm(default_local_llm="lmstudio")
                        else:
                            llm = setup_llm(default_local_llm="ollama-mistral")
                        if not llm:
                            return
                        
                        # Formatando as informações dos arquivos
                        files_info_str = "\n\n".join([
                            f"Arquivo: {info.filename} ({info.file_type})\nColunas: {', '.join(info.columns)}\nAmostra de dados:\n{info.sample_data}"
                            for info in st.session_state.files_info
                        ])
                        
                        # Obter histórico da conversa
                        history = st.session_state.memory.load_memory_variables({})
                        
                        # Criar prompt template com memória
                        prompt = ChatPromptTemplate.from_template("""
                        Histórico da conversa:
                        {history}

                        Você é um especialista em análise de dados financeiros. Analise os seguintes arquivos:

                        {files_info}

                        Dados calculados:
                        - Estado com mais vendas: {estado_top} com {total_estado} em {count_estado} notas
                        - Cidade com mais vendas: {cidade_top} com {total_cidade} em {count_cidade} notas
                        - Data com mais vendas: {data_top} com {total_data} em {count_data} notas
                        - Produto mais vendido: '{produto_top}' com {quantidade_produto} unidades (faturamento: {faturamento_produto})
                        - Produto mais caro: '{produto_caro}' com valor unitário de {valor_produto}
                        - Total geral: {total_geral} em {total_notas} notas fiscais

                        Para perguntas específicas:
                        - Para valores por estado: filtre por ESTADO_DESTINO
                        - Para valores por cidade: filtre por CIDADE_DESTINO
                        - Para produtos: consulte DESCRIÇÃO DO PRODUTO/SERVIÇO
                        - Use sempre os dados calculados quando possível

                        Pergunta: {question}

                        Responda de forma clara, direta e precisa, com valores formatados.
                        Inclua sempre a quantidade de notas fiscais quando relevante.
                        Se não encontrar dados, informe claramente.
                        """)
                        
                        # Criar a cadeia de processamento
                        chain = prompt | llm | StrOutputParser()
                        
                        # Executar
                        llm_response = chain.invoke({
                            "history": history['history'],
                            "files_info": files_info_str,
                            "estado_top": st.session_state.metrics.get('estado_mais_vendeu', 'Nenhum'),
                            "total_estado": format_currency(st.session_state.metrics.get('total_por_estado', {}).get(st.session_state.metrics.get('estado_mais_vendeu', ''), 0)),
                            "count_estado": st.session_state.metrics.get('contagem_por_estado', {}).get(st.session_state.metrics.get('estado_mais_vendeu', ''), 0),
                            "cidade_top": st.session_state.metrics.get('cidade_mais_vendeu', 'Nenhuma'),
                            "total_cidade": format_currency(st.session_state.metrics.get('total_por_cidade', {}).get(st.session_state.metrics.get('cidade_mais_vendeu', ''), 0)),
                            "count_cidade": st.session_state.metrics.get('contagem_por_cidade', {}).get(st.session_state.metrics.get('cidade_mais_vendeu', ''), 0),
                            "data_top": st.session_state.metrics.get('data_mais_vendas', pd.Timestamp('today')).strftime('%d/%m/%Y'),
                            "total_data": format_currency(st.session_state.metrics.get('total_por_data', {}).get(st.session_state.metrics.get('data_mais_vendas', ''), 0)),
                            "count_data": st.session_state.metrics.get('contagem_por_data', {}).get(st.session_state.metrics.get('data_mais_vendas', ''), 0),
                            "produto_top": st.session_state.metrics.get('produto_mais_vendido', 'Nenhum'),
                            "quantidade_produto": int(st.session_state.metrics.get('quantidade_produto', {}).get(st.session_state.metrics.get('produto_mais_vendido', ''), 0)),
                            "faturamento_produto": format_currency(st.session_state.metrics.get('faturamento_produto', {}).get(st.session_state.metrics.get('produto_mais_vendido', ''), 0)),
                            "produto_caro": st.session_state.metrics.get('produto_mais_caro', 'Nenhum'),
                            "valor_produto": format_currency(st.session_state.metrics.get('valor_produto', {}).get(st.session_state.metrics.get('produto_mais_caro', ''), 0)),
                            "total_geral": format_currency(st.session_state.metrics.get('total_vendas', 0)),
                            "total_notas": st.session_state.metrics.get('total_notas', 0),
                            "question": user_question
                        })
                        response = llm_response
                    
                    # Adicionar resposta ao histórico e memória
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.session_state.memory.save_context({"input": user_question}, {"output": response})
                    
                    with st.chat_message("assistant"):
                        st.markdown(response)
                        
                except Exception as e:
                    error_msg = f"Desculpe, ocorreu um erro ao processar sua pergunta: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": error_msg
                    })

if __name__ == "__main__":
    main()