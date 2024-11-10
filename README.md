# Hackathon Meta

Este repositório contém o código desenvolvido para o *Hackathon Meta*, um projeto que integra tecnologias de Inteligência Artificial e bancos de dados vetoriais para criar um assistente interativo capaz de identificar e recomendar oportunidades de desenvolvimento profissional personalizadas.

## Índice

1. [Visão Geral do Projeto](#visão-geral-do-projeto)
2. [Estrutura do Projeto](#estrutura-do-projeto)
   - [Backend CLI (chatbot.py)](#backend-cli-chatbotpy)
   - [Backend com Frontend (aplicativo.py)](#backend-com-frontend-aplicativopy)
   - [Interface Frontend (index.html)](#interface-frontend-indexhtml)
   - [Crew AI (crew.py)](#crew-ai-crewpy)
   - [Arquivo Principal (main.py)](#arquivo-principal-mainpy)
   - [Configurações de Agentes (agents.yaml)](#configurações-de-agentes-agentsyaml)
   - [Configurações de Tarefas (tasks.yaml)](#configurações-de-tarefas-tasksyaml)
   - [Script de Limpeza do Banco de Dados (apaga_base_TESTE.py)](#script-de-limpeza-do-banco-de-dados-apaga_base_testepy)
3. [Instalação e Configuração](#instalação-e-configuração)
4. [Como Executar](#como-executar)
   - [Executando o Backend CLI](#executando-o-backend-cli)
   - [Executando o Backend com Frontend](#executando-o-backend-com-frontend)
   - [Executando o Frontend](#executando-o-frontend)
   - [Executando o Crew AI](#executando-o-crew-ai)
   - [Utilizando o Script de Limpeza](#utilizando-o-script-de-limpeza)
5. [Estrutura do Código](#estrutura-do-código)
   - [chatbot.py](#chatbotpy)
   - [aplicativo.py](#aplicativopy)
   - [index.html](#indexhtml)
   - [crew.py](#crewpy)
   - [main.py](#mainpy)
   - [agents.yaml](#agentsyaml)
   - [tasks.yaml](#tasksyaml)
   - [apaga_base_TESTE.py](#apaga_base_testepy)
6. [Contribuição](#contribuição)
7. [Licença](#licença)

---

## Visão Geral do Projeto

O projeto visa criar um assistente virtual interativo que utiliza modelos de linguagem avançados e bancos de dados vetoriais para fornecer recomendações personalizadas de desenvolvimento profissional aos usuários. O assistente interage com o usuário, coleta informações relevantes e utiliza agentes de IA para processar os dados e gerar insights valiosos.

## Estrutura do Projeto

O projeto é composto por vários componentes principais, cada um desempenhando um papel crucial na funcionalidade geral do sistema.

### Backend CLI (chatbot.py)

Este é o backend que permite interações com o assistente diretamente via terminal. Ele utiliza várias APIs e bibliotecas para gerenciar as conversas e armazenar informações no MongoDB.

#### Principais Funcionalidades

- *Groq API*: Utilizada para geração de respostas inteligentes, conectada via chave de API.
- *Cohere API*: Cria embeddings para armazenar informações contextuais do usuário no MongoDB Atlas.
- *MongoDB Atlas*: Banco de dados vetorial onde as conversas e o contexto são salvos e consultados.
- *Gestão de Memória Conversacional*: Armazena o histórico de mensagens e utiliza um buffer de memória para manter o contexto do usuário.
- *Validação de Contexto*: Verifica se as informações coletadas são suficientes para gerar recomendações personalizadas.
- *Execução de Agentes Crew AI*: Integração com agentes que processam as informações coletadas para gerar recomendações ao usuário.

### Backend com Frontend (aplicativo.py)

Este backend foi desenvolvido para permitir a integração com uma interface web. Ele expõe endpoints via Flask que facilitam a comunicação com o assistente interativo e com o banco de dados MongoDB.

#### Principais Funcionalidades

- *Endpoints RESTful*: Disponibiliza rotas para interagir com o assistente, permitindo que os usuários façam login, enviem mensagens e recebam respostas.
- *Cross-Origin Resource Sharing (CORS)*: Configurado para aceitar requisições de diferentes origens, facilitando a integração com aplicações frontend.
- *MongoDB Atlas e Vector Search*: Armazena o contexto e histórico das conversas, permitindo buscas eficientes com embeddings de documentos.
- *Validação de Contexto e Recomendação*: Verifica se o contexto possui informações suficientes antes de acionar os agentes Crew AI para recomendações.

### Interface Frontend (index.html)

A interface do usuário foi construída em HTML, CSS (Tailwind CSS) e JavaScript, proporcionando uma experiência de interação fluida e responsiva com o assistente virtual.

#### Principais Funcionalidades

- *Login e Logout*: Formulário de autenticação com validação e tratamento de erros para garantir que apenas usuários autorizados acessem a plataforma.
- *Interface de Chat*: Uma experiência de chat intuitiva, com bolhas de diálogo para mensagens do usuário e do assistente.
- *Exibição de Recomendações*: Um painel de oportunidades e recomendações profissionais que são carregadas e exibidas dinamicamente com base nas interações do usuário.
- *Sidebar*: Menu lateral com opções de navegação, incluindo links para o chat e para recomendações.
- *Compatibilidade com Backend Flask*: Interage com o backend via chamadas AJAX para envio de mensagens e carregamento de dados.

### Crew AI (crew.py)

Este arquivo configura os agentes e tarefas do Crew AI, responsáveis por processar as informações do usuário e encontrar oportunidades de desenvolvimento profissional.

#### Descrição

O crew.py define a classe OportunityFinderCrew, que utiliza a biblioteca Crew AI para criar agentes especializados em diferentes áreas, como análise de contexto do usuário, busca de oportunidades de emprego, eventos, cursos e desenvolvimento profissional.

- *Agentes Definidos*:
  - user_context_analyzer: Coleta e analisa o contexto do usuário a partir do banco de dados.
  - job_opportunity_finder: Encontra oportunidades de emprego relevantes.
  - event_opportunity_finder: Descobre eventos que contribuem para o desenvolvimento pessoal e profissional.
  - course_opportunity_finder: Encontra cursos que atendam às necessidades educacionais do usuário.
  - professional_development_finder: Identifica oportunidades de desenvolvimento profissional.

### Arquivo Principal (main.py)

Este é o script que serve para rodar o Crew AI. Ele executa todas as tarefas definidas no crew.py e mostra as oportunidades encontradas.

#### Descrição

O main.py inicializa uma instância de OportunityFinderCrew e executa o método kickoff para iniciar o processamento das tarefas. Após a execução, ele exibe as oportunidades salvas no MongoDB.

### Configurações de Agentes (agents.yaml)

Este arquivo define as configurações dos agentes utilizados pelo Crew AI.

#### Descrição

O agents.yaml especifica o papel, objetivo e histórico de cada agente, fornecendo contexto para que eles possam executar suas tarefas de forma eficaz.

### Configurações de Tarefas (tasks.yaml)

Este arquivo define as tarefas que serão executadas pelos agentes.

#### Descrição

O tasks.yaml descreve as tarefas, incluindo uma descrição detalhada e o output esperado, garantindo que cada agente saiba exatamente o que precisa ser feito.

### Script de Limpeza do Banco de Dados (apaga_base_TESTE.py)

Este script é responsável por limpar as coleções de contexto e histórico de conversas no MongoDB, permitindo reiniciar a memória do assistente virtual para começar novas interações do zero.

#### Descrição

O apaga_base_TESTE.py conecta-se ao MongoDB usando as credenciais fornecidas, solicita uma confirmação de segurança ao usuário e, em seguida, remove todos os documentos das coleções Contexto e HistoricoConversa.

## Instalação e Configuração

1. *Clonar o Repositório*

   bash
   git clone https://github.com/seu_usuario/hackathon_meta.git
   cd hackathon_meta
   

2. *Criar e Ativar Ambiente Virtual*

   bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate  # Windows
   

3. *Instalar Dependências*

   bash
   pip install -r requirements.txt
   

4. *Configurar Variáveis de Ambiente*

   Crie um arquivo .env na raiz do projeto com as seguintes informações:

   ini
   GROQ_API_KEY=sua_groq_api_key
   COHERE_API_KEY=sua_cohere_api_key
   MONGODB_USERNAME=seu_mongodb_username
   MONGODB_PASSWORD=seu_mongodb_password
   

## Como Executar

### Executando o Backend CLI

Para iniciar o assistente via terminal:

bash
python chatbot.py


### Executando o Backend com Frontend

Para iniciar o servidor Flask:

bash
python aplicativo.py


O backend estará disponível em http://localhost:5000.

### Executando o Frontend

Após iniciar o backend Flask, abra o arquivo index.html em um navegador web moderno para interagir com a interface do usuário.

### Executando o Crew AI

Para executar o Crew AI e processar as oportunidades:

bash
python main.py


Você pode especificar um user_id:

bash
python main.py seu_user_id


### Utilizando o Script de Limpeza

Para limpar o banco de dados MongoDB:

bash
python apaga_base_TESTE.py


Confirme a operação digitando "s" quando solicitado.

## Estrutura do Código

### chatbot.py

Este arquivo permite interações com o assistente diretamente via terminal.

#### Funções Principais

- **carregar_memoria()**: Carrega a memória de conversas anteriores do MongoDB.
- **salvar_memoria()**: Salva o histórico da conversa atual no banco de dados.
- **gerar_resposta_groq()**: Gera respostas usando o modelo da Groq API.
- **armazenar_mensagem_no_vectorstore()**: Armazena mensagens do usuário no MongoDB Atlas Vector Search.
- **validar_contexto_suficiente()**: Verifica se todas as informações necessárias foram coletadas.
- **acionar_agentes()**: Aciona os agentes do Crew AI para processar as informações e gerar recomendações.
- **detectar_intencao_ai()**: Detecta a intenção do usuário em receber recomendações.
- **iniciar_conversa()**: Gerencia o fluxo da conversa com o usuário.

### aplicativo.py

Este arquivo é o backend desenvolvido para integração com a interface web.

#### Endpoints Disponíveis

- **/login**: Autentica o usuário.
- **/conversa**: Carrega o histórico da conversa do usuário.
- **/mensagem**: Envia uma mensagem do usuário ao chatbot e recebe a resposta.
- **/oportunidades**: Retorna oportunidades de desenvolvimento personalizadas para o usuário.

### index.html

Este arquivo é a interface frontend do projeto.

#### Componentes Principais

- *Cabeçalho Fixo*: Título "Elevate" sempre visível.
- *Sidebar*: Menu de navegação com opções para chat, recomendações e logout.
- *Formulário de Login*: Autenticação de usuários.
- *Área de Chat*: Interação com o assistente virtual.
- *Painel de Recomendações*: Exibição dinâmica de oportunidades personalizadas.

#### Scripts JavaScript

- **sendMessage()**: Envia mensagens e processa respostas.
- **toggleSidebar()**: Controla a visibilidade da sidebar.
- **showChat(), showRecommendations()**: Navegação entre as seções.
- **loadConversation(), loadOpportunities()**: Carrega histórico e oportunidades do backend.

### crew.py

Este arquivo configura os agentes e tarefas do Crew AI.

#### Descrição

Define a classe OportunityFinderCrew, que utiliza a biblioteca Crew AI para criar agentes especializados e tarefas associadas.

- *Agentes*:
  - **user_context_analyzer**: Analisa o contexto do usuário.
  - **job_opportunity_finder**: Encontra oportunidades de emprego.
  - **event_opportunity_finder**: Encontra eventos relevantes.
  - **course_opportunity_finder**: Encontra cursos educacionais.
  - **professional_development_finder**: Encontra oportunidades de desenvolvimento profissional.

- *Tarefas*:
  - **analyze_user_context_task**: Tarefa para analisar o contexto do usuário.
  - **find_job_opportunities_task**: Tarefa para encontrar oportunidades de emprego.
  - **find_event_opportunities_task**: Tarefa para encontrar eventos.
  - **find_course_opportunities_task**: Tarefa para encontrar cursos.
  - **find_professional_development_task**: Tarefa para encontrar oportunidades de desenvolvimento profissional.

### main.py

Este é o script principal que executa o Crew AI.

#### Descrição

- **run()**: Função que executa o Crew AI e exibe as oportunidades encontradas.
- **mostrar_oportunidades(user_id)**: Exibe as oportunidades salvas no MongoDB para o user_id especificado.

### agents.yaml

Este arquivo define as configurações dos agentes utilizados pelo Crew AI.

#### Descrição

Cada agente é configurado com:

- **role**: O papel do agente.
- **goal**: O objetivo do agente.
- **backstory**: Contexto adicional para o agente.

### tasks.yaml

Este arquivo define as tarefas que serão executadas pelos agentes.

#### Descrição

Cada tarefa inclui:

- **description**: Descrição detalhada da tarefa.
- **expected_output**: O output esperado após a conclusão da tarefa.

### apaga_base_TESTE.py

Script para limpar as coleções de contexto e histórico de conversas no MongoDB.

#### Descrição

- Conecta ao MongoDB usando as credenciais fornecidas.
- Solicita confirmação do usuário antes de prosseguir com a exclusão.
- Remove todos os documentos das coleções Contexto e HistoricoConversa.

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests para melhorias, correções de bugs ou novas funcionalidades.

## Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE).
