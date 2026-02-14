# CodeArchaeologist - System Design Document

## Executive Summary

CodeArchaeologist is a distributed AI-powered platform that combines traditional static analysis with modern ML techniques (contrastive learning, embeddings, RAG) to provide intelligent code understanding, documentation, and knowledge management. The system follows a microservices architecture with clear separation between ingestion, ML processing, API services, and frontend.

## 1. Architecture Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  React Web   │  │  VS Code     │  │  JetBrains   │         │
│  │     UI       │  │  Extension   │  │    Plugin    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTPS/WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         API Gateway                             │
│              (FastAPI + Node.js Express)                        │
│  Authentication │ Rate Limiting │ Request Routing               │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────┐
│  Code Ingestion  │ │  ML Pipeline │ │  Knowledge   │
│     Service      │ │   Service    │ │   Service    │
│                  │ │              │ │              │
│ • Repo Scanner   │ │ • Embedding  │ │ • AI Chat    │
│ • Parser         │ │ • Training   │ │ • Doc Gen    │
│ • AST Builder    │ │ • Inference  │ │ • Search     │
└──────────────────┘ └──────────────┘ └──────────────┘
        │                    │                 │
        └────────────────────┼─────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Data Layer                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │PostgreSQL│  │ ChromaDB │  │  Neo4j   │  │  Redis   │       │
│  │(Metadata)│  │ (Vectors)│  │ (Graph)  │  │ (Cache)  │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              S3/GCS (Code & Artifacts)               │      │
│  └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Technology Stack Summary

| Layer | Technologies |
|-------|-------------|
| Frontend | React 18, TypeScript, Tailwind CSS, D3.js, Mermaid.js, React Query |
| API Gateway | FastAPI (Python), Express (Node.js), Nginx |
| Services | Python 3.11+, Node.js 20+, FastAPI, Celery |
| ML/AI | PyTorch, GraphCodeBERT, sentence-transformers, LangChain, Claude API |
| Parsing | Tree-sitter, Pygments, NetworkX, ast (Python stdlib) |
| Databases | PostgreSQL 15, ChromaDB, Neo4j 5, Redis 7 |
| Storage | AWS S3 / Google Cloud Storage |
| Message Queue | RabbitMQ / Redis |
| Monitoring | Prometheus, Grafana, Sentry |
| DevOps | Docker, Kubernetes, GitHub Actions, Terraform |


## 2. Component Breakdown

### 2.1 Code Ingestion Service

**Responsibilities:**
- Clone and scan Git repositories
- Detect programming languages and file types
- Parse code into AST, CFG, DFG representations
- Extract metadata: functions, classes, imports, comments
- Build dependency graphs
- Detect dead code and code smells

**Key Modules:**

```python
# Repository Scanner
class RepositoryScanner:
    def clone_repository(self, git_url: str, branch: str) -> Path
    def scan_files(self, repo_path: Path) -> List[CodeFile]
    def detect_language(self, file_path: Path) -> Language
    def extract_metadata(self, file_path: Path) -> FileMetadata

# Multi-Language Parser
class CodeParser:
    def parse_to_ast(self, code: str, language: Language) -> AST
    def extract_functions(self, ast: AST) -> List[Function]
    def extract_classes(self, ast: AST) -> List[Class]
    def extract_imports(self, ast: AST) -> List[Import]
    def build_cfg(self, ast: AST) -> ControlFlowGraph
    def build_dfg(self, ast: AST) -> DataFlowGraph

# Dependency Analyzer
class DependencyAnalyzer:
    def build_dependency_graph(self, files: List[CodeFile]) -> nx.DiGraph
    def detect_circular_dependencies(self, graph: nx.DiGraph) -> List[Cycle]
    def calculate_coupling_metrics(self, graph: nx.DiGraph) -> Dict

# Dead Code Detector
class DeadCodeDetector:
    def find_unused_functions(self, graph: nx.DiGraph) -> List[Function]
    def find_unused_imports(self, files: List[CodeFile]) -> List[Import]
    def calculate_confidence_score(self, item: CodeItem) -> float
```

**Data Flow:**
1. User provides Git URL → Clone repository
2. Scan all files → Detect languages
3. Parse each file → Generate AST/CFG/DFG
4. Extract entities → Store in PostgreSQL
5. Build dependency graph → Store in Neo4j
6. Trigger ML pipeline for embedding generation

### 2.2 ML Pipeline Service

**Responsibilities:**
- Generate code embeddings using GraphCodeBERT
- Train contrastive learning models (SimCLR)
- Perform semantic code search
- Cluster related code
- Detect cross-language clones

**Key Modules:**

```python
# Embedding Generator
class EmbeddingGenerator:
    def __init__(self, model_name: str = "microsoft/graphcodebert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    def generate_embedding(self, code: str, language: str) -> np.ndarray
    def batch_generate(self, codes: List[str]) -> np.ndarray
    def store_embeddings(self, embeddings: np.ndarray, ids: List[str])

# Contrastive Learning Trainer
class ContrastiveLearner:
    def create_positive_pairs(self, code: str) -> List[Tuple[str, str]]
    def create_negative_pairs(self, codes: List[str]) -> List[Tuple[str, str]]
    def nt_xent_loss(self, z_i: Tensor, z_j: Tensor, temperature: float) -> Tensor
    def train_epoch(self, dataloader: DataLoader) -> float
    def evaluate(self, test_data: Dataset) -> Dict[str, float]

# Semantic Search Engine
class SemanticSearchEngine:
    def __init__(self, vector_db: ChromaDB):
        self.vector_db = vector_db
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]
    def search_by_embedding(self, embedding: np.ndarray, top_k: int) -> List[SearchResult]
    def filter_by_language(self, results: List[SearchResult], lang: str) -> List[SearchResult]
    def explain_match(self, query: str, result: SearchResult) -> str

# Code Clustering
class CodeClusterer:
    def cluster_embeddings(self, embeddings: np.ndarray, method: str = "hdbscan") -> np.ndarray
    def label_clusters(self, clusters: np.ndarray, codes: List[str]) -> Dict[int, str]
    def visualize_clusters(self, embeddings: np.ndarray, labels: np.ndarray) -> Figure

# Cross-Language Detector
class CrossLanguageDetector:
    def find_similar_across_languages(self, code: str, source_lang: str) -> List[Match]
    def calculate_semantic_similarity(self, code1: str, code2: str) -> float
```

**ML Pipeline Architecture:**

```
Input Code → Preprocessing → Multi-View Generation
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
                  AST View      Text View       Semantic View
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                          GraphCodeBERT Encoder
                                    ▼
                          768-dim Embedding
                                    ▼
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              ChromaDB Store   Contrastive      Clustering
                              Learning Model
```

### 2.3 Knowledge Service

**Responsibilities:**
- AI-powered documentation generation
- Tribal knowledge capture via chat
- Living documentation management
- Technical debt tracking
- Bug pattern recognition

**Key Modules:**

```python
# Documentation Generator
class DocumentationGenerator:
    def __init__(self, llm_client: ClaudeClient):
        self.llm = llm_client
        self.rag_engine = RAGEngine()
    
    def generate_readme(self, repo: Repository) -> str
    def generate_function_docstring(self, function: Function) -> str
    def generate_architecture_diagram(self, repo: Repository) -> str
    def update_stale_docs(self, changed_files: List[str]) -> List[str]

# AI Interview System
class AIInterviewer:
    def start_interview(self, developer: Developer, context: str) -> Interview
    def ask_question(self, interview: Interview) -> str
    def process_answer(self, answer: str) -> KnowledgeItem
    def extract_structured_knowledge(self, conversation: List[Message]) -> List[KnowledgeItem]
    def link_to_code(self, knowledge: KnowledgeItem, repo: Repository) -> List[CodeEntity]

# Technical Debt Tracker
class TechDebtTracker:
    def detect_code_smells(self, file: CodeFile) -> List[CodeSmell]
    def calculate_severity(self, smell: CodeSmell) -> int
    def estimate_fix_time(self, smell: CodeSmell) -> timedelta
    def prioritize_debt(self, items: List[TechDebtItem]) -> List[TechDebtItem]
    def track_trends(self, repo: Repository, time_range: DateRange) -> TrendData

# Bug Pattern Recognizer
class BugPatternRecognizer:
    def learn_from_incident(self, bug: Bug, fix_commit: Commit)
    def extract_pattern(self, bug: Bug) -> Pattern
    def scan_for_patterns(self, code: str) -> List[PatternMatch]
    def generate_warning(self, match: PatternMatch) -> Warning
```

### 2.4 API Gateway

**Responsibilities:**
- Authentication and authorization
- Rate limiting and throttling
- Request routing to services
- Response caching
- API documentation (OpenAPI/Swagger)

**Endpoints:**

```python
# Repository Management
POST   /api/v1/repositories                    # Add repository
GET    /api/v1/repositories                    # List repositories
GET    /api/v1/repositories/{id}               # Get repository details
DELETE /api/v1/repositories/{id}               # Remove repository
POST   /api/v1/repositories/{id}/scan          # Trigger scan

# Search
GET    /api/v1/search?q={query}&lang={lang}    # Semantic search
POST   /api/v1/search/similar                  # Find similar code
GET    /api/v1/search/suggestions              # Search suggestions

# Documentation
GET    /api/v1/docs/{repo_id}/readme           # Get README
POST   /api/v1/docs/{repo_id}/generate         # Generate docs
GET    /api/v1/docs/{repo_id}/stale            # Get stale docs
PUT    /api/v1/docs/{file_id}                  # Update documentation

# Knowledge
POST   /api/v1/knowledge/interview             # Start interview
POST   /api/v1/knowledge/items                 # Add knowledge item
GET    /api/v1/knowledge/items                 # List knowledge
GET    /api/v1/knowledge/export                # Export as Markdown

# Technical Debt
GET    /api/v1/debt/{repo_id}                  # Get tech debt items
GET    /api/v1/debt/{repo_id}/trends           # Get debt trends
POST   /api/v1/debt/{item_id}/resolve          # Mark as resolved

# Onboarding
POST   /api/v1/onboarding/learning-path        # Generate learning path
GET    /api/v1/onboarding/progress/{user_id}   # Get progress
POST   /api/v1/onboarding/track                # Track activity

# Analytics
GET    /api/v1/analytics/dependencies/{repo_id} # Dependency graph
GET    /api/v1/analytics/clusters/{repo_id}     # Code clusters
GET    /api/v1/analytics/metrics/{repo_id}      # Code metrics

# Security
GET    /api/v1/security/scan/{repo_id}          # Security scan results
GET    /api/v1/security/secrets/{repo_id}       # Detected secrets
GET    /api/v1/security/licenses/{repo_id}      # License report
```

### 2.5 Frontend Application

**Component Structure:**

```
src/
├── components/
│   ├── common/
│   │   ├── Button.tsx
│   │   ├── Input.tsx
│   │   ├── Modal.tsx
│   │   └── Spinner.tsx
│   ├── search/
│   │   ├── SearchBar.tsx
│   │   ├── SearchResults.tsx
│   │   ├── CodeSnippet.tsx
│   │   └── FilterPanel.tsx
│   ├── repository/
│   │   ├── RepoList.tsx
│   │   ├── RepoDetails.tsx
│   │   ├── DependencyGraph.tsx (D3.js)
│   │   └── FileTree.tsx
│   ├── documentation/
│   │   ├── DocViewer.tsx
│   │   ├── DocEditor.tsx
│   │   ├── ArchitectureDiagram.tsx (Mermaid.js)
│   │   └── StaleDocsBadge.tsx
│   ├── knowledge/
│   │   ├── InterviewChat.tsx
│   │   ├── KnowledgeList.tsx
│   │   ├── KnowledgeGraph.tsx
│   │   └── ExportButton.tsx
│   ├── debt/
│   │   ├── DebtDashboard.tsx
│   │   ├── DebtList.tsx
│   │   ├── DebtTrends.tsx (Chart.js)
│   │   └── PriorityMatrix.tsx
│   └── onboarding/
│       ├── LearningPath.tsx
│       ├── ProgressTracker.tsx
│       ├── InteractiveTutorial.tsx
│       └── QuizComponent.tsx
├── pages/
│   ├── Dashboard.tsx
│   ├── Search.tsx
│   ├── Repository.tsx
│   ├── Documentation.tsx
│   ├── Knowledge.tsx
│   ├── TechDebt.tsx
│   ├── Onboarding.tsx
│   └── Settings.tsx
├── hooks/
│   ├── useSearch.ts
│   ├── useRepository.ts
│   ├── useDocumentation.ts
│   └── useAuth.ts
├── services/
│   ├── api.ts
│   ├── auth.ts
│   └── websocket.ts
└── utils/
    ├── formatting.ts
    ├── validation.ts
    └── constants.ts
```


## 3. Data Models

### 3.1 PostgreSQL Schema

```sql
-- Repositories
CREATE TABLE repositories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    git_url TEXT NOT NULL,
    branch VARCHAR(100) DEFAULT 'main',
    last_scanned_at TIMESTAMP,
    total_files INTEGER,
    total_lines INTEGER,
    primary_language VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Code Files
CREATE TABLE code_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repository_id UUID REFERENCES repositories(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    language VARCHAR(50),
    lines_of_code INTEGER,
    complexity_score FLOAT,
    last_modified TIMESTAMP,
    git_commit_hash VARCHAR(40),
    is_dead_code BOOLEAN DEFAULT FALSE,
    dead_code_confidence FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(repository_id, file_path)
);

-- Functions
CREATE TABLE functions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id UUID REFERENCES code_files(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    signature TEXT,
    start_line INTEGER,
    end_line INTEGER,
    cyclomatic_complexity INTEGER,
    parameters JSONB,
    return_type VARCHAR(100),
    docstring TEXT,
    is_public BOOLEAN,
    is_tested BOOLEAN,
    embedding_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Classes
CREATE TABLE classes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id UUID REFERENCES code_files(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    methods JSONB,
    properties JSONB,
    inheritance JSONB,
    docstring TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Dependencies
CREATE TABLE dependencies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_file_id UUID REFERENCES code_files(id) ON DELETE CASCADE,
    target_file_id UUID REFERENCES code_files(id) ON DELETE CASCADE,
    import_type VARCHAR(50),
    is_circular BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(source_file_id, target_file_id)
);

-- Documentation
CREATE TABLE documentation (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repository_id UUID REFERENCES repositories(id) ON DELETE CASCADE,
    file_id UUID REFERENCES code_files(id) ON DELETE SET NULL,
    doc_type VARCHAR(50), -- 'readme', 'function', 'class', 'architecture'
    content TEXT NOT NULL,
    format VARCHAR(20) DEFAULT 'markdown',
    is_stale BOOLEAN DEFAULT FALSE,
    staleness_score FLOAT,
    generated_at TIMESTAMP DEFAULT NOW(),
    last_updated TIMESTAMP DEFAULT NOW(),
    version INTEGER DEFAULT 1
);

-- Knowledge Items
CREATE TABLE knowledge_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repository_id UUID REFERENCES repositories(id) ON DELETE CASCADE,
    author_id UUID REFERENCES users(id),
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(50), -- 'architecture', 'bug', 'workaround', 'optimization', 'security'
    tags TEXT[],
    linked_files JSONB,
    linked_functions JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Technical Debt
CREATE TABLE tech_debt_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repository_id UUID REFERENCES repositories(id) ON DELETE CASCADE,
    file_id UUID REFERENCES code_files(id) ON DELETE CASCADE,
    debt_type VARCHAR(50), -- 'code_smell', 'security', 'performance', 'maintainability'
    title VARCHAR(255) NOT NULL,
    description TEXT,
    severity INTEGER CHECK (severity BETWEEN 1 AND 5),
    estimated_hours FLOAT,
    impact_score FLOAT,
    effort_score FLOAT,
    priority_score FLOAT,
    status VARCHAR(20) DEFAULT 'open', -- 'open', 'in_progress', 'resolved', 'wont_fix'
    detected_at TIMESTAMP DEFAULT NOW(),
    resolved_at TIMESTAMP
);

-- Bug Patterns
CREATE TABLE bug_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_name VARCHAR(255) NOT NULL,
    pattern_signature TEXT,
    description TEXT,
    language VARCHAR(50),
    severity INTEGER,
    example_code TEXT,
    fix_suggestion TEXT,
    occurrences INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Users
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'developer', -- 'admin', 'developer', 'viewer'
    avatar_url TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP
);

-- Learning Paths
CREATE TABLE learning_paths (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    repository_id UUID REFERENCES repositories(id) ON DELETE CASCADE,
    recommended_files JSONB,
    recommended_concepts JSONB,
    progress JSONB,
    estimated_completion_days INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Audit Logs
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    details JSONB,
    ip_address INET,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_code_files_repo ON code_files(repository_id);
CREATE INDEX idx_code_files_language ON code_files(language);
CREATE INDEX idx_functions_file ON functions(file_id);
CREATE INDEX idx_functions_name ON functions(name);
CREATE INDEX idx_knowledge_repo ON knowledge_items(repository_id);
CREATE INDEX idx_knowledge_category ON knowledge_items(category);
CREATE INDEX idx_debt_repo ON tech_debt_items(repository_id);
CREATE INDEX idx_debt_status ON tech_debt_items(status);
CREATE INDEX idx_debt_priority ON tech_debt_items(priority_score DESC);
CREATE INDEX idx_audit_user ON audit_logs(user_id);
CREATE INDEX idx_audit_created ON audit_logs(created_at DESC);
```

### 3.2 ChromaDB Collections

```python
# Code Embeddings Collection
code_embeddings = {
    "name": "code_embeddings",
    "metadata": {
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 200,
        "hnsw:M": 16
    },
    "documents": [
        {
            "id": "func_uuid_1",
            "embedding": [0.123, 0.456, ...],  # 768-dim vector
            "metadata": {
                "repository_id": "repo_uuid",
                "file_path": "src/utils.py",
                "function_name": "validate_email",
                "language": "python",
                "code_snippet": "def validate_email(email: str) -> bool: ...",
                "indexed_at": "2024-01-15T10:30:00Z"
            }
        }
    ]
}

# Documentation Embeddings Collection
doc_embeddings = {
    "name": "doc_embeddings",
    "metadata": {"hnsw:space": "cosine"},
    "documents": [
        {
            "id": "doc_uuid_1",
            "embedding": [0.789, 0.012, ...],
            "metadata": {
                "repository_id": "repo_uuid",
                "doc_type": "readme",
                "title": "Authentication System",
                "content": "The authentication system uses JWT tokens...",
                "indexed_at": "2024-01-15T10:30:00Z"
            }
        }
    ]
}

# Knowledge Embeddings Collection
knowledge_embeddings = {
    "name": "knowledge_embeddings",
    "metadata": {"hnsw:space": "cosine"},
    "documents": [
        {
            "id": "knowledge_uuid_1",
            "embedding": [0.345, 0.678, ...],
            "metadata": {
                "repository_id": "repo_uuid",
                "category": "architecture",
                "title": "Why we chose microservices",
                "author": "senior_dev@company.com",
                "created_at": "2024-01-15T10:30:00Z"
            }
        }
    ]
}
```

### 3.3 Neo4j Graph Schema

```cypher
// Node Types
(:Repository {id, name, git_url, language})
(:File {id, path, language, lines})
(:Function {id, name, signature, complexity})
(:Class {id, name, methods})
(:Developer {id, name, email})
(:Commit {id, hash, message, timestamp})
(:Bug {id, title, severity})
(:Decision {id, title, rationale})
(:Concept {id, name, description})

// Relationship Types
(:File)-[:BELONGS_TO]->(:Repository)
(:Function)-[:DEFINED_IN]->(:File)
(:Class)-[:DEFINED_IN]->(:File)
(:File)-[:IMPORTS]->(:File)
(:File)-[:DEPENDS_ON]->(:File)
(:Function)-[:CALLS]->(:Function)
(:Class)-[:INHERITS]->(:Class)
(:Developer)-[:AUTHORED]->(:Commit)
(:Commit)-[:MODIFIED]->(:File)
(:Bug)-[:FOUND_IN]->(:File)
(:Bug)-[:FIXED_BY]->(:Commit)
(:Decision)-[:AFFECTS]->(:File)
(:Developer)-[:KNOWS]->(:Concept)
(:Function)-[:IMPLEMENTS]->(:Concept)

// Example Queries
// Find all files that depend on a specific file
MATCH (f:File {path: 'src/auth.py'})<-[:DEPENDS_ON]-(dependent:File)
RETURN dependent.path

// Find developers who worked on authentication
MATCH (d:Developer)-[:AUTHORED]->(c:Commit)-[:MODIFIED]->(f:File)
WHERE f.path CONTAINS 'auth'
RETURN DISTINCT d.name, COUNT(c) as commits
ORDER BY commits DESC

// Find circular dependencies
MATCH path = (f1:File)-[:DEPENDS_ON*]->(f1)
WHERE length(path) > 1
RETURN path

// Find bug patterns
MATCH (b:Bug)-[:FOUND_IN]->(f:File)<-[:DEFINED_IN]-(func:Function)
WHERE b.severity >= 4
RETURN func.name, COUNT(b) as bug_count
ORDER BY bug_count DESC
```

### 3.4 Redis Cache Schema

```python
# Cache Keys Structure
cache_keys = {
    # Search Results Cache (TTL: 1 hour)
    "search:{query_hash}": {
        "results": [...],
        "total": 42,
        "cached_at": "2024-01-15T10:30:00Z"
    },
    
    # Repository Metadata Cache (TTL: 5 minutes)
    "repo:{repo_id}:metadata": {
        "name": "my-project",
        "total_files": 1234,
        "languages": {"python": 60, "javascript": 40}
    },
    
    # User Session Cache (TTL: 24 hours)
    "session:{session_id}": {
        "user_id": "user_uuid",
        "role": "developer",
        "permissions": [...]
    },
    
    # Rate Limiting (TTL: 1 minute)
    "ratelimit:{user_id}:{endpoint}": {
        "count": 45,
        "reset_at": "2024-01-15T10:31:00Z"
    },
    
    # Job Queue Status
    "job:{job_id}": {
        "status": "processing",
        "progress": 65,
        "started_at": "2024-01-15T10:25:00Z"
    }
}
```


## 4. API Contracts

### 4.1 REST API Specifications

```yaml
openapi: 3.0.0
info:
  title: CodeArchaeologist API
  version: 1.0.0
  description: AI-powered code documentation and knowledge management platform

servers:
  - url: https://api.codearchaeologist.com/v1
    description: Production server
  - url: http://localhost:8000/v1
    description: Development server

components:
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
  
  schemas:
    Repository:
      type: object
      properties:
        id:
          type: string
          format: uuid
        name:
          type: string
        git_url:
          type: string
        branch:
          type: string
        last_scanned_at:
          type: string
          format: date-time
        total_files:
          type: integer
        primary_language:
          type: string
    
    SearchResult:
      type: object
      properties:
        id:
          type: string
        code_snippet:
          type: string
        file_path:
          type: string
        language:
          type: string
        similarity_score:
          type: number
          format: float
        context:
          type: string
        highlights:
          type: array
          items:
            type: string
    
    KnowledgeItem:
      type: object
      properties:
        id:
          type: string
          format: uuid
        title:
          type: string
        content:
          type: string
        category:
          type: string
          enum: [architecture, bug, workaround, optimization, security]
        tags:
          type: array
          items:
            type: string
        linked_files:
          type: array
          items:
            type: string
        author:
          type: object
          properties:
            id:
              type: string
            name:
              type: string
        created_at:
          type: string
          format: date-time
    
    TechDebtItem:
      type: object
      properties:
        id:
          type: string
          format: uuid
        title:
          type: string
        description:
          type: string
        debt_type:
          type: string
        severity:
          type: integer
          minimum: 1
          maximum: 5
        estimated_hours:
          type: number
        priority_score:
          type: number
        status:
          type: string
          enum: [open, in_progress, resolved, wont_fix]
        file_path:
          type: string
        detected_at:
          type: string
          format: date-time

paths:
  /repositories:
    post:
      summary: Add a new repository
      security:
        - BearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - git_url
              properties:
                git_url:
                  type: string
                  example: "https://github.com/user/repo.git"
                branch:
                  type: string
                  default: "main"
                auto_scan:
                  type: boolean
                  default: true
      responses:
        '201':
          description: Repository added successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Repository'
        '400':
          description: Invalid request
        '401':
          description: Unauthorized
    
    get:
      summary: List all repositories
      security:
        - BearerAuth: []
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            default: 1
        - name: limit
          in: query
          schema:
            type: integer
            default: 20
        - name: language
          in: query
          schema:
            type: string
      responses:
        '200':
          description: List of repositories
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/Repository'
                  total:
                    type: integer
                  page:
                    type: integer
                  limit:
                    type: integer

  /search:
    get:
      summary: Semantic code search
      security:
        - BearerAuth: []
      parameters:
        - name: q
          in: query
          required: true
          schema:
            type: string
          example: "function that validates email addresses"
        - name: language
          in: query
          schema:
            type: string
        - name: repository_id
          in: query
          schema:
            type: string
            format: uuid
        - name: limit
          in: query
          schema:
            type: integer
            default: 10
      responses:
        '200':
          description: Search results
          content:
            application/json:
              schema:
                type: object
                properties:
                  results:
                    type: array
                    items:
                      $ref: '#/components/schemas/SearchResult'
                  total:
                    type: integer
                  query_time_ms:
                    type: number

  /knowledge/items:
    post:
      summary: Create knowledge item
      security:
        - BearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - title
                - content
                - category
              properties:
                title:
                  type: string
                content:
                  type: string
                category:
                  type: string
                tags:
                  type: array
                  items:
                    type: string
                linked_files:
                  type: array
                  items:
                    type: string
      responses:
        '201':
          description: Knowledge item created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/KnowledgeItem'

  /debt/{repository_id}:
    get:
      summary: Get technical debt items
      security:
        - BearerAuth: []
      parameters:
        - name: repository_id
          in: path
          required: true
          schema:
            type: string
            format: uuid
        - name: status
          in: query
          schema:
            type: string
            enum: [open, in_progress, resolved, wont_fix]
        - name: sort_by
          in: query
          schema:
            type: string
            enum: [priority, severity, estimated_hours]
            default: priority
      responses:
        '200':
          description: List of technical debt items
          content:
            application/json:
              schema:
                type: object
                properties:
                  items:
                    type: array
                    items:
                      $ref: '#/components/schemas/TechDebtItem'
                  total:
                    type: integer
                  summary:
                    type: object
                    properties:
                      total_estimated_hours:
                        type: number
                      by_severity:
                        type: object
```

### 4.2 WebSocket Events

```typescript
// Client -> Server Events
interface ClientEvents {
  // Real-time search
  'search:start': {
    query: string;
    filters: SearchFilters;
  };
  
  // AI Interview
  'interview:start': {
    repository_id: string;
    context?: string;
  };
  'interview:answer': {
    interview_id: string;
    answer: string;
  };
  
  // Progress tracking
  'progress:subscribe': {
    job_id: string;
  };
}

// Server -> Client Events
interface ServerEvents {
  // Search results streaming
  'search:result': {
    result: SearchResult;
    is_final: boolean;
  };
  
  // Interview questions
  'interview:question': {
    interview_id: string;
    question: string;
    context: string;
  };
  'interview:complete': {
    interview_id: string;
    knowledge_items: KnowledgeItem[];
  };
  
  // Job progress
  'progress:update': {
    job_id: string;
    status: 'queued' | 'processing' | 'completed' | 'failed';
    progress: number;
    message: string;
  };
  
  // Notifications
  'notification': {
    type: 'info' | 'warning' | 'error' | 'success';
    title: string;
    message: string;
  };
}
```


## 5. ML Pipeline Design

### 5.1 SimCLR Contrastive Learning Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input: Code Snippet                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Multi-View Augmentation                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   AST    │  │   CFG    │  │   DFG    │  │   Text   │       │
│  │   View   │  │   View   │  │   View   │  │   View   │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              GraphCodeBERT Encoder (Shared Weights)             │
│                    Input: Token Sequences                       │
│                    Output: 768-dim Embeddings                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
         Embedding z_i   Embedding z_j   Embedding z_k
         (AST View)      (CFG View)      (Text View)
                │             │             │
                └─────────────┼─────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Projection Head (MLP)                        │
│              768-dim → 512-dim → 256-dim                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NT-Xent Loss Function                        │
│                                                                 │
│  L = -log( exp(sim(z_i, z_j) / τ) /                           │
│            Σ exp(sim(z_i, z_k) / τ) )                         │
│                                                                 │
│  where:                                                         │
│  - sim(u, v) = cosine_similarity(u, v)                        │
│  - τ = temperature parameter (0.07)                            │
│  - z_i, z_j = positive pair (same code, different views)      │
│  - z_k = negative samples (different code)                     │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Training Pipeline

```python
class ContrastiveTrainingPipeline:
    def __init__(self, config: TrainingConfig):
        self.encoder = GraphCodeBERT.from_pretrained("microsoft/graphcodebert-base")
        self.projection_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.temperature = 0.07
        self.optimizer = AdamW(self.parameters(), lr=1e-5)
        
    def create_augmentations(self, code: str) -> List[str]:
        """Generate multiple views of the same code"""
        views = []
        
        # View 1: AST representation
        ast = self.parse_to_ast(code)
        views.append(self.ast_to_sequence(ast))
        
        # View 2: CFG representation
        cfg = self.build_cfg(code)
        views.append(self.cfg_to_sequence(cfg))
        
        # View 3: DFG representation
        dfg = self.build_dfg(code)
        views.append(self.dfg_to_sequence(dfg))
        
        # View 4: Raw text with minor transformations
        views.append(self.normalize_code(code))
        
        # View 5: Semantic tokens
        views.append(self.extract_semantic_tokens(code))
        
        return views
    
    def nt_xent_loss(self, z_i: Tensor, z_j: Tensor, batch_size: int) -> Tensor:
        """Normalized Temperature-scaled Cross Entropy Loss"""
        # Concatenate positive pairs
        z = torch.cat([z_i, z_j], dim=0)  # Shape: (2*batch_size, 256)
        
        # Compute similarity matrix
        sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_matrix = sim_matrix / self.temperature
        
        # Create mask for positive pairs
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        mask = mask.roll(shifts=batch_size, dims=0)
        
        # Compute loss
        exp_sim = torch.exp(sim_matrix)
        exp_sim = exp_sim.masked_fill(~mask, 0)
        
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = -log_prob.masked_select(mask).mean()
        
        return loss
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.encoder.train()
        self.projection_head.train()
        total_loss = 0
        
        for batch in dataloader:
            codes = batch['code']
            
            # Generate augmentations
            views_i = [self.create_augmentations(code)[0] for code in codes]
            views_j = [self.create_augmentations(code)[1] for code in codes]
            
            # Encode views
            z_i = self.encode(views_i)
            z_j = self.encode(views_j)
            
            # Project to lower dimension
            z_i = self.projection_head(z_i)
            z_j = self.projection_head(z_j)
            
            # Compute loss
            loss = self.nt_xent_loss(z_i, z_j, len(codes))
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def encode(self, texts: List[str]) -> Tensor:
        """Encode text to embeddings"""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        outputs = self.encoder(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS token
```

### 5.3 Inference Pipeline

```python
class InferencePipeline:
    def __init__(self, model_path: str):
        self.encoder = GraphCodeBERT.from_pretrained(model_path)
        self.encoder.eval()
        self.vector_db = ChromaDB(collection_name="code_embeddings")
        
    @torch.no_grad()
    def generate_embedding(self, code: str, language: str) -> np.ndarray:
        """Generate embedding for a code snippet"""
        # Preprocess code
        code = self.preprocess_code(code, language)
        
        # Tokenize
        inputs = self.tokenizer(
            code,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Generate embedding
        outputs = self.encoder(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding[0]
    
    def search(self, query: str, top_k: int = 10, filters: Dict = None) -> List[SearchResult]:
        """Semantic search for code"""
        # Generate query embedding
        query_embedding = self.generate_embedding(query, language="natural")
        
        # Search in vector database
        results = self.vector_db.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )
        
        # Format results
        search_results = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            search_results.append(SearchResult(
                id=metadata['id'],
                code_snippet=doc,
                file_path=metadata['file_path'],
                language=metadata['language'],
                similarity_score=1 - distance,  # Convert distance to similarity
                rank=i + 1
            ))
        
        return search_results
    
    def find_similar_code(self, code: str, language: str, threshold: float = 0.8) -> List[Match]:
        """Find similar code across the codebase"""
        embedding = self.generate_embedding(code, language)
        
        results = self.vector_db.query(
            query_embeddings=[embedding],
            n_results=50
        )
        
        # Filter by threshold
        matches = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            similarity = 1 - distance
            if similarity >= threshold:
                matches.append(Match(
                    code=doc,
                    file_path=metadata['file_path'],
                    language=metadata['language'],
                    similarity=similarity
                ))
        
        return matches
```

### 5.4 Model Training Configuration

```yaml
training:
  model:
    base_model: "microsoft/graphcodebert-base"
    embedding_dim: 768
    projection_dim: 256
    
  hyperparameters:
    batch_size: 32
    learning_rate: 1e-5
    weight_decay: 0.01
    temperature: 0.07
    epochs: 10
    warmup_steps: 1000
    
  augmentation:
    num_views: 5
    view_types:
      - ast
      - cfg
      - dfg
      - text
      - semantic
    
  data:
    train_split: 0.8
    val_split: 0.1
    test_split: 0.1
    max_code_length: 512
    min_code_length: 10
    
  optimization:
    optimizer: "adamw"
    scheduler: "cosine"
    gradient_clip: 1.0
    mixed_precision: true
    
  evaluation:
    metrics:
      - code_clone_detection_f1
      - semantic_search_ndcg
      - cross_language_accuracy
    eval_frequency: 500  # steps
    
  checkpointing:
    save_frequency: 1000  # steps
    keep_best_n: 3
    metric_to_track: "val_loss"
```


## 6. System Workflows

### 6.1 Repository Onboarding Flow

```
User adds repository
        │
        ▼
┌───────────────────┐
│ Clone Repository  │
│ (Git Service)     │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Scan Files       │
│  Detect Languages │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Parse Code       │
│  (Tree-sitter)    │
│  • AST            │
│  • Functions      │
│  • Classes        │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Store Metadata    │
│ (PostgreSQL)      │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Build Graphs      │
│ • Dependencies    │
│ • Call Graph      │
│ (Neo4j)           │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Generate          │
│ Embeddings        │
│ (ML Pipeline)     │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Store Embeddings  │
│ (ChromaDB)        │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Generate Docs     │
│ (LLM + RAG)       │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Detect Tech Debt  │
│ & Security Issues │
└───────────────────┘
        │
        ▼
    Complete
```

### 6.2 Semantic Search Flow

```
User enters query
        │
        ▼
┌───────────────────┐
│ Preprocess Query  │
│ • Tokenize        │
│ • Normalize       │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Check Cache       │
│ (Redis)           │
└───────────────────┘
        │
    ┌───┴───┐
    │       │
  Hit     Miss
    │       │
    │       ▼
    │  ┌───────────────────┐
    │  │ Generate Embedding│
    │  │ (GraphCodeBERT)   │
    │  └───────────────────┘
    │       │
    │       ▼
    │  ┌───────────────────┐
    │  │ Vector Search     │
    │  │ (ChromaDB)        │
    │  │ • Cosine Sim      │
    │  │ • Top-K           │
    │  └───────────────────┘
    │       │
    │       ▼
    │  ┌───────────────────┐
    │  │ Apply Filters     │
    │  │ • Language        │
    │  │ • Repository      │
    │  │ • Date Range      │
    │  └───────────────────┘
    │       │
    │       ▼
    │  ┌───────────────────┐
    │  │ Enrich Results    │
    │  │ (PostgreSQL)      │
    │  │ • File metadata   │
    │  │ • Context         │
    │  └───────────────────┘
    │       │
    │       ▼
    │  ┌───────────────────┐
    │  │ Cache Results     │
    │  │ (Redis)           │
    │  └───────────────────┘
    │       │
    └───────┘
        │
        ▼
┌───────────────────┐
│ Return Results    │
│ • Ranked          │
│ • Highlighted     │
│ • Explained       │
└───────────────────┘
```

### 6.3 AI Interview Flow

```
Developer starts interview
        │
        ▼
┌───────────────────┐
│ Analyze Context   │
│ • Recent commits  │
│ • Complex files   │
│ • Tech debt       │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Generate Question │
│ (Claude API)      │
│ • Targeted        │
│ • Contextual      │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Present Question  │
│ (WebSocket)       │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Receive Answer    │
│ (User Input)      │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Extract Knowledge │
│ (NLP + LLM)       │
│ • Entities        │
│ • Relationships   │
│ • Categories      │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Link to Code      │
│ • Files           │
│ • Functions       │
│ • Commits         │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Store Knowledge   │
│ (PostgreSQL)      │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Generate Embedding│
│ (ChromaDB)        │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Update Graph      │
│ (Neo4j)           │
└───────────────────┘
        │
        ▼
    More questions?
    │           │
   Yes         No
    │           │
    └───────────┘
        │
        ▼
┌───────────────────┐
│ Export Summary    │
│ (Markdown)        │
└───────────────────┘
```

### 6.4 Living Documentation Update Flow

```
Git commit/push
        │
        ▼
┌───────────────────┐
│ Git Hook Trigger  │
│ (post-commit)     │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Detect Changed    │
│ Files             │
│ (git diff)        │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Queue Update Job  │
│ (Celery/RabbitMQ) │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Re-parse Changed  │
│ Files             │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Update Embeddings │
│ (Incremental)     │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Check Related Docs│
│ • Function docs   │
│ • README          │
│ • Architecture    │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Calculate         │
│ Staleness Score   │
│ • Time delta      │
│ • Change severity │
└───────────────────┘
        │
        ▼
    Stale?
    │    │
   Yes   No
    │    │
    ▼    └──────┐
┌───────────────────┐
│ Flag as Stale     │
│ (PostgreSQL)      │
└───────────────────┘
    │              │
    ▼              │
┌───────────────────┐
│ Notify Developer  │
│ (Email/Slack)     │
└───────────────────┘
    │              │
    ▼              │
┌───────────────────┐
│ Auto-regenerate?  │
│ (if configured)   │
└───────────────────┘
    │              │
    └──────────────┘
        │
        ▼
    Complete
```

### 6.5 Technical Debt Detection Flow

```
Code scan triggered
        │
        ▼
┌───────────────────┐
│ Static Analysis   │
│ • Complexity      │
│ • Code smells     │
│ • Duplication     │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ ML-based Detection│
│ • Pattern matching│
│ • Anomaly detect  │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Security Scan     │
│ • Secrets         │
│ • Vulnerabilities │
│ • OWASP patterns  │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Calculate Scores  │
│ • Severity        │
│ • Impact          │
│ • Effort          │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Estimate Fix Time │
│ (ML model)        │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Prioritize Items  │
│ • Priority score  │
│ • Business value  │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Store Debt Items  │
│ (PostgreSQL)      │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Update Dashboard  │
│ • Trends          │
│ • Metrics         │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Generate Alerts   │
│ (if thresholds    │
│  exceeded)        │
└───────────────────┘
```


## 7. Deployment Architecture

### 7.1 Docker Compose (Development)

```yaml
version: '3.8'

services:
  # API Gateway
  api-gateway:
    build: ./services/api-gateway
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/codearch
      - REDIS_URL=redis://redis:6379
      - RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672
    depends_on:
      - postgres
      - redis
      - rabbitmq
    volumes:
      - ./services/api-gateway:/app
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

  # Code Ingestion Service
  ingestion-service:
    build: ./services/ingestion
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/codearch
      - NEO4J_URI=bolt://neo4j:7687
      - S3_BUCKET=codearch-repos
    depends_on:
      - postgres
      - neo4j
    volumes:
      - ./services/ingestion:/app
      - repo-cache:/tmp/repos

  # ML Pipeline Service
  ml-service:
    build: ./services/ml-pipeline
    environment:
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8001
      - MODEL_PATH=/models
    depends_on:
      - chromadb
    volumes:
      - ./services/ml-pipeline:/app
      - ml-models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Knowledge Service
  knowledge-service:
    build: ./services/knowledge
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/codearch
      - CLAUDE_API_KEY=${CLAUDE_API_KEY}
      - CHROMA_HOST=chromadb
    depends_on:
      - postgres
      - chromadb
    volumes:
      - ./services/knowledge:/app

  # Celery Workers
  celery-worker:
    build: ./services/api-gateway
    command: celery -A tasks worker --loglevel=info --concurrency=4
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/codearch
      - REDIS_URL=redis://redis:6379
      - RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672
    depends_on:
      - postgres
      - redis
      - rabbitmq
    volumes:
      - ./services/api-gateway:/app

  # Frontend
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
      - REACT_APP_WS_URL=ws://localhost:8000
    volumes:
      - ./frontend:/app
      - /app/node_modules
    command: npm start

  # PostgreSQL
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=codearch
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data

  # ChromaDB
  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma-data:/chroma/chroma
    environment:
      - ALLOW_RESET=TRUE

  # Neo4j
  neo4j:
    image: neo4j:5-community
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["apoc", "graph-data-science"]
    volumes:
      - neo4j-data:/data

  # Redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  # RabbitMQ
  rabbitmq:
    image: rabbitmq:3-management-alpine
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      - RABBITMQ_DEFAULT_USER=guest
      - RABBITMQ_DEFAULT_PASS=guest
    volumes:
      - rabbitmq-data:/var/lib/rabbitmq

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  # Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards

volumes:
  postgres-data:
  chroma-data:
  neo4j-data:
  redis-data:
  rabbitmq-data:
  prometheus-data:
  grafana-data:
  ml-models:
  repo-cache:
```

### 7.2 Kubernetes (Production)

```yaml
# Namespace
apiVersion: v1
kind: Namespace
metadata:
  name: codearchaeologist

---
# API Gateway Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
  namespace: codearchaeologist
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
    spec:
      containers:
      - name: api-gateway
        image: codearch/api-gateway:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        - name: REDIS_URL
          value: redis://redis-service:6379
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
# API Gateway Service
apiVersion: v1
kind: Service
metadata:
  name: api-gateway-service
  namespace: codearchaeologist
spec:
  selector:
    app: api-gateway
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
# ML Service Deployment (with GPU)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-service
  namespace: codearchaeologist
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-service
  template:
    metadata:
      labels:
        app: ml-service
    spec:
      containers:
      - name: ml-service
        image: codearch/ml-service:latest
        ports:
        - containerPort: 8002
        env:
        - name: CHROMA_HOST
          value: chromadb-service
        - name: MODEL_PATH
          value: /models
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4000m"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: ml-models-pvc

---
# PostgreSQL StatefulSet
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: codearchaeologist
spec:
  serviceName: postgres-service
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: codearch
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi

---
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-gateway-hpa
  namespace: codearchaeologist
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-gateway
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80

---
# Ingress
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: codearch-ingress
  namespace: codearchaeologist
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.codearchaeologist.com
    secretName: codearch-tls
  rules:
  - host: api.codearchaeologist.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-gateway-service
            port:
              number: 80
```

### 7.3 Infrastructure as Code (Terraform)

```hcl
# AWS Infrastructure
provider "aws" {
  region = "us-west-2"
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "codearch-vpc"
  }
}

# EKS Cluster
resource "aws_eks_cluster" "main" {
  name     = "codearch-cluster"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = "1.28"

  vpc_config {
    subnet_ids = aws_subnet.private[*].id
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy
  ]
}

# EKS Node Group
resource "aws_eks_node_group" "main" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "codearch-nodes"
  node_role_arn   = aws_iam_role.eks_nodes.arn
  subnet_ids      = aws_subnet.private[*].id

  scaling_config {
    desired_size = 3
    max_size     = 10
    min_size     = 2
  }

  instance_types = ["t3.xlarge"]

  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy
  ]
}

# GPU Node Group for ML
resource "aws_eks_node_group" "gpu" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "codearch-gpu-nodes"
  node_role_arn   = aws_iam_role.eks_nodes.arn
  subnet_ids      = aws_subnet.private[*].id

  scaling_config {
    desired_size = 1
    max_size     = 3
    min_size     = 1
  }

  instance_types = ["g4dn.xlarge"]

  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy
  ]
}

# RDS PostgreSQL
resource "aws_db_instance" "main" {
  identifier           = "codearch-db"
  engine              = "postgres"
  engine_version      = "15.3"
  instance_class      = "db.r6g.xlarge"
  allocated_storage   = 100
  storage_type        = "gp3"
  storage_encrypted   = true
  
  db_name  = "codearch"
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "mon:04:00-mon:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "codearch-final-snapshot"
  
  tags = {
    Name = "codearch-db"
  }
}

# ElastiCache Redis
resource "aws_elasticache_cluster" "main" {
  cluster_id           = "codearch-redis"
  engine              = "redis"
  engine_version      = "7.0"
  node_type           = "cache.r6g.large"
  num_cache_nodes     = 1
  parameter_group_name = "default.redis7"
  port                = 6379
  
  subnet_group_name    = aws_elasticache_subnet_group.main.name
  security_group_ids   = [aws_security_group.redis.id]
  
  tags = {
    Name = "codearch-redis"
  }
}

# S3 Bucket for Code Storage
resource "aws_s3_bucket" "repos" {
  bucket = "codearch-repositories"
  
  tags = {
    Name = "codearch-repos"
  }
}

resource "aws_s3_bucket_versioning" "repos" {
  bucket = aws_s3_bucket.repos.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "repos" {
  bucket = aws_s3_bucket.repos.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "main" {
  name              = "/aws/eks/codearch-cluster"
  retention_in_days = 30
}

# Outputs
output "cluster_endpoint" {
  value = aws_eks_cluster.main.endpoint
}

output "db_endpoint" {
  value = aws_db_instance.main.endpoint
}

output "redis_endpoint" {
  value = aws_elasticache_cluster.main.cache_nodes[0].address
}
```


## 8. Security Architecture

### 8.1 Authentication & Authorization

```python
# JWT Token Structure
{
  "header": {
    "alg": "RS256",
    "typ": "JWT"
  },
  "payload": {
    "sub": "user_uuid",
    "email": "user@example.com",
    "role": "developer",
    "permissions": ["read:repos", "write:knowledge", "admin:users"],
    "iat": 1640000000,
    "exp": 1640086400
  }
}

# RBAC Permissions Matrix
PERMISSIONS = {
    "admin": [
        "read:*",
        "write:*",
        "delete:*",
        "admin:*"
    ],
    "developer": [
        "read:repos",
        "read:docs",
        "read:knowledge",
        "read:debt",
        "write:knowledge",
        "write:docs",
        "scan:repos"
    ],
    "viewer": [
        "read:repos",
        "read:docs",
        "read:knowledge",
        "read:debt"
    ]
}

# Middleware for Authorization
class AuthorizationMiddleware:
    def __init__(self, required_permission: str):
        self.required_permission = required_permission
    
    async def __call__(self, request: Request, call_next):
        # Extract JWT from Authorization header
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        
        try:
            # Verify and decode token
            payload = jwt.decode(token, PUBLIC_KEY, algorithms=["RS256"])
            
            # Check permission
            user_permissions = payload.get("permissions", [])
            if not self.has_permission(user_permissions, self.required_permission):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            # Add user info to request state
            request.state.user = payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        return await call_next(request)
    
    def has_permission(self, user_perms: List[str], required: str) -> bool:
        # Check for wildcard permissions
        if "*" in user_perms or "read:*" in user_perms:
            return True
        
        # Check for exact match
        if required in user_perms:
            return True
        
        # Check for resource wildcard (e.g., "write:*" covers "write:repos")
        resource_type = required.split(":")[0]
        if f"{resource_type}:*" in user_perms:
            return True
        
        return False
```

### 8.2 Data Encryption

```python
# Encryption at Rest
class EncryptionService:
    def __init__(self, kms_client):
        self.kms = kms_client
        self.key_id = os.getenv("KMS_KEY_ID")
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data using AWS KMS"""
        response = self.kms.encrypt(
            KeyId=self.key_id,
            Plaintext=data.encode()
        )
        return base64.b64encode(response['CiphertextBlob']).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        ciphertext = base64.b64decode(encrypted_data)
        response = self.kms.decrypt(
            CiphertextBlob=ciphertext
        )
        return response['Plaintext'].decode()

# Field-Level Encryption for Database
class EncryptedField:
    def __init__(self, encryption_service: EncryptionService):
        self.encryption = encryption_service
    
    def __set__(self, instance, value):
        if value is not None:
            encrypted = self.encryption.encrypt_sensitive_data(value)
            instance.__dict__[self.name] = encrypted
    
    def __get__(self, instance, owner):
        encrypted = instance.__dict__.get(self.name)
        if encrypted is not None:
            return self.encryption.decrypt_sensitive_data(encrypted)
        return None

# Usage in Models
class Repository(Base):
    __tablename__ = "repositories"
    
    id = Column(UUID, primary_key=True)
    name = Column(String)
    git_url = Column(String)
    access_token = Column(String)  # Encrypted in application layer
    
    def __init__(self, encryption_service: EncryptionService):
        self._encryption = encryption_service
    
    @property
    def access_token(self):
        return self._encryption.decrypt_sensitive_data(self._access_token)
    
    @access_token.setter
    def access_token(self, value):
        self._access_token = self._encryption.encrypt_sensitive_data(value)
```

### 8.3 Secret Detection

```python
# Secret Scanner
class SecretScanner:
    def __init__(self):
        self.patterns = {
            "aws_access_key": r"AKIA[0-9A-Z]{16}",
            "aws_secret_key": r"[0-9a-zA-Z/+]{40}",
            "github_token": r"ghp_[0-9a-zA-Z]{36}",
            "slack_token": r"xox[baprs]-[0-9a-zA-Z-]{10,72}",
            "private_key": r"-----BEGIN (RSA|DSA|EC|OPENSSH) PRIVATE KEY-----",
            "jwt": r"eyJ[A-Za-z0-9-_=]+\.eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_.+/=]+",
            "generic_api_key": r"['\"]?api[_-]?key['\"]?\s*[:=]\s*['\"]?[0-9a-zA-Z]{32,}['\"]?"
        }
        
        self.entropy_threshold = 4.5
    
    def scan_code(self, code: str, file_path: str) -> List[SecretMatch]:
        """Scan code for potential secrets"""
        matches = []
        
        # Pattern-based detection
        for secret_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, code):
                matches.append(SecretMatch(
                    type=secret_type,
                    value=match.group(),
                    line=code[:match.start()].count('\n') + 1,
                    file_path=file_path,
                    confidence=0.9
                ))
        
        # Entropy-based detection
        for line_num, line in enumerate(code.split('\n'), 1):
            # Look for high-entropy strings
            for word in re.findall(r'["\']([A-Za-z0-9+/=]{20,})["\']', line):
                entropy = self.calculate_entropy(word)
                if entropy > self.entropy_threshold:
                    matches.append(SecretMatch(
                        type="high_entropy_string",
                        value=word,
                        line=line_num,
                        file_path=file_path,
                        confidence=min(entropy / 6.0, 1.0)
                    ))
        
        return matches
    
    def calculate_entropy(self, string: str) -> float:
        """Calculate Shannon entropy of a string"""
        if not string:
            return 0
        
        entropy = 0
        for char in set(string):
            prob = string.count(char) / len(string)
            entropy -= prob * math.log2(prob)
        
        return entropy
```

### 8.4 Rate Limiting

```python
# Rate Limiter using Redis
class RateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def check_rate_limit(
        self,
        user_id: str,
        endpoint: str,
        max_requests: int = 100,
        window_seconds: int = 60
    ) -> Tuple[bool, Dict]:
        """Check if user has exceeded rate limit"""
        key = f"ratelimit:{user_id}:{endpoint}"
        
        # Get current count
        current = await self.redis.get(key)
        
        if current is None:
            # First request in window
            await self.redis.setex(key, window_seconds, 1)
            return True, {
                "remaining": max_requests - 1,
                "reset_at": time.time() + window_seconds
            }
        
        current = int(current)
        
        if current >= max_requests:
            # Rate limit exceeded
            ttl = await self.redis.ttl(key)
            return False, {
                "remaining": 0,
                "reset_at": time.time() + ttl
            }
        
        # Increment counter
        await self.redis.incr(key)
        ttl = await self.redis.ttl(key)
        
        return True, {
            "remaining": max_requests - current - 1,
            "reset_at": time.time() + ttl
        }

# Rate Limiting Middleware
class RateLimitMiddleware:
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
    
    async def __call__(self, request: Request, call_next):
        user_id = request.state.user.get("sub")
        endpoint = f"{request.method}:{request.url.path}"
        
        # Different limits for different endpoints
        limits = {
            "GET:/api/v1/search": (100, 60),  # 100 requests per minute
            "POST:/api/v1/repositories": (10, 3600),  # 10 per hour
            "default": (1000, 3600)  # 1000 per hour
        }
        
        max_requests, window = limits.get(endpoint, limits["default"])
        
        allowed, info = await self.rate_limiter.check_rate_limit(
            user_id, endpoint, max_requests, window
        )
        
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(info["reset_at"]))
                }
            )
        
        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(int(info["reset_at"]))
        
        return response
```

### 8.5 Input Validation & Sanitization

```python
# Input Validation
from pydantic import BaseModel, validator, Field
from typing import Optional

class RepositoryCreate(BaseModel):
    git_url: str = Field(..., regex=r"^https?://.*\.git$")
    branch: str = Field(default="main", max_length=100)
    auto_scan: bool = True
    
    @validator("git_url")
    def validate_git_url(cls, v):
        # Prevent SSRF attacks
        parsed = urlparse(v)
        if parsed.hostname in ["localhost", "127.0.0.1", "0.0.0.0"]:
            raise ValueError("Local URLs are not allowed")
        
        # Only allow specific domains in production
        if os.getenv("ENV") == "production":
            allowed_domains = ["github.com", "gitlab.com", "bitbucket.org"]
            if not any(parsed.hostname.endswith(domain) for domain in allowed_domains):
                raise ValueError(f"Only {allowed_domains} are allowed")
        
        return v

class SearchQuery(BaseModel):
    q: str = Field(..., min_length=1, max_length=500)
    language: Optional[str] = Field(None, regex=r"^[a-z]+$")
    repository_id: Optional[str] = Field(None, regex=r"^[0-9a-f-]{36}$")
    limit: int = Field(default=10, ge=1, le=100)
    
    @validator("q")
    def sanitize_query(cls, v):
        # Remove potentially dangerous characters
        v = re.sub(r"[<>\"']", "", v)
        return v.strip()

# SQL Injection Prevention (using parameterized queries)
class RepositoryService:
    def __init__(self, db: Session):
        self.db = db
    
    def search_repositories(self, name: str) -> List[Repository]:
        # GOOD: Parameterized query
        return self.db.query(Repository).filter(
            Repository.name.ilike(f"%{name}%")
        ).all()
        
        # BAD: String concatenation (vulnerable to SQL injection)
        # query = f"SELECT * FROM repositories WHERE name LIKE '%{name}%'"
        # return self.db.execute(query).fetchall()
```


## 9. Monitoring & Observability

### 9.1 Metrics Collection

```yaml
# Prometheus Configuration
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # API Gateway Metrics
  - job_name: 'api-gateway'
    static_configs:
      - targets: ['api-gateway:8000']
    metrics_path: '/metrics'
  
  # ML Service Metrics
  - job_name: 'ml-service'
    static_configs:
      - targets: ['ml-service:8002']
  
  # PostgreSQL Metrics
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
  
  # Redis Metrics
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
  
  # Node Metrics
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

# Alert Rules
groups:
  - name: codearch_alerts
    interval: 30s
    rules:
      # High Error Rate
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors/sec"
      
      # High Response Time
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "P95 response time is {{ $value }}s"
      
      # Database Connection Pool Exhausted
      - alert: DatabasePoolExhausted
        expr: db_connection_pool_active / db_connection_pool_max > 0.9
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Database connection pool nearly exhausted"
      
      # High Memory Usage
      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage on {{ $labels.instance }}"
      
      # ML Model Inference Slow
      - alert: SlowMLInference
        expr: histogram_quantile(0.95, rate(ml_inference_duration_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "ML inference is slow"
```

### 9.2 Application Metrics

```python
# Custom Metrics using Prometheus Client
from prometheus_client import Counter, Histogram, Gauge, Info

# Request Metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

# Search Metrics
search_queries_total = Counter(
    'search_queries_total',
    'Total search queries',
    ['language', 'repository']
)

search_results_count = Histogram(
    'search_results_count',
    'Number of search results returned',
    buckets=[0, 1, 5, 10, 20, 50, 100]
)

search_duration_seconds = Histogram(
    'search_duration_seconds',
    'Search query duration',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# ML Metrics
ml_inference_duration_seconds = Histogram(
    'ml_inference_duration_seconds',
    'ML model inference duration',
    ['model_name', 'operation']
)

ml_embedding_generation_total = Counter(
    'ml_embedding_generation_total',
    'Total embeddings generated',
    ['model_name']
)

# Repository Metrics
repositories_scanned_total = Counter(
    'repositories_scanned_total',
    'Total repositories scanned'
)

code_files_processed_total = Counter(
    'code_files_processed_total',
    'Total code files processed',
    ['language']
)

# Knowledge Metrics
knowledge_items_created_total = Counter(
    'knowledge_items_created_total',
    'Total knowledge items created',
    ['category']
)

interviews_completed_total = Counter(
    'interviews_completed_total',
    'Total AI interviews completed'
)

# Technical Debt Metrics
tech_debt_items_detected = Gauge(
    'tech_debt_items_detected',
    'Current number of tech debt items',
    ['severity', 'status']
)

# Database Metrics
db_connection_pool_active = Gauge(
    'db_connection_pool_active',
    'Active database connections'
)

db_connection_pool_max = Gauge(
    'db_connection_pool_max',
    'Maximum database connections'
)

db_query_duration_seconds = Histogram(
    'db_query_duration_seconds',
    'Database query duration',
    ['query_type']
)

# Cache Metrics
cache_hits_total = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type']
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type']
)

# Middleware for automatic metrics collection
class MetricsMiddleware:
    async def __call__(self, request: Request, call_next):
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            http_requests_total.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            http_request_duration_seconds.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            http_requests_total.labels(
                method=request.method,
                endpoint=request.url.path,
                status=500
            ).inc()
            raise
```

### 9.3 Logging Strategy

```python
# Structured Logging Configuration
import structlog
from pythonjsonlogger import jsonlogger

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Usage Examples
class SearchService:
    def search(self, query: str, user_id: str):
        logger.info(
            "search_started",
            query=query,
            user_id=user_id,
            query_length=len(query)
        )
        
        try:
            results = self._perform_search(query)
            
            logger.info(
                "search_completed",
                query=query,
                user_id=user_id,
                result_count=len(results),
                duration_ms=duration
            )
            
            return results
            
        except Exception as e:
            logger.error(
                "search_failed",
                query=query,
                user_id=user_id,
                error=str(e),
                exc_info=True
            )
            raise

# Log Levels and Usage
# DEBUG: Detailed diagnostic information
logger.debug("embedding_generated", embedding_dim=768, code_length=150)

# INFO: General informational messages
logger.info("repository_scanned", repo_id=repo_id, files_count=1234)

# WARNING: Warning messages for potentially harmful situations
logger.warning("stale_documentation_detected", file_path=path, days_old=30)

# ERROR: Error messages for serious problems
logger.error("ml_inference_failed", model=model_name, error=str(e))

# CRITICAL: Critical messages for very serious errors
logger.critical("database_connection_lost", attempts=retry_count)
```

### 9.4 Distributed Tracing

```python
# OpenTelemetry Configuration
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor

# Setup tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Configure Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# Instrument frameworks
FastAPIInstrumentor.instrument_app(app)
SQLAlchemyInstrumentor().instrument(engine=engine)
RedisInstrumentor().instrument()

# Manual tracing for custom operations
class SearchService:
    def search(self, query: str):
        with tracer.start_as_current_span("search_operation") as span:
            span.set_attribute("query", query)
            span.set_attribute("query_length", len(query))
            
            # Generate embedding
            with tracer.start_as_current_span("generate_embedding"):
                embedding = self.generate_embedding(query)
                span.set_attribute("embedding_dim", len(embedding))
            
            # Vector search
            with tracer.start_as_current_span("vector_search"):
                results = self.vector_db.search(embedding)
                span.set_attribute("results_count", len(results))
            
            # Enrich results
            with tracer.start_as_current_span("enrich_results"):
                enriched = self.enrich_results(results)
            
            span.set_attribute("final_results_count", len(enriched))
            return enriched
```

### 9.5 Grafana Dashboards

```json
{
  "dashboard": {
    "title": "CodeArchaeologist - Overview",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "{{endpoint}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Response Time (P95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "{{endpoint}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Search Performance",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(search_duration_seconds_bucket[5m]))",
            "legendFormat": "P95"
          },
          {
            "expr": "histogram_quantile(0.50, rate(search_duration_seconds_bucket[5m]))",
            "legendFormat": "P50"
          }
        ],
        "type": "graph"
      },
      {
        "title": "ML Inference Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(ml_inference_duration_seconds_bucket[5m]))",
            "legendFormat": "{{model_name}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Database Connection Pool",
        "targets": [
          {
            "expr": "db_connection_pool_active",
            "legendFormat": "Active"
          },
          {
            "expr": "db_connection_pool_max",
            "legendFormat": "Max"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m]))",
            "legendFormat": "{{cache_type}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Technical Debt Items",
        "targets": [
          {
            "expr": "tech_debt_items_detected",
            "legendFormat": "Severity {{severity}}"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```


## 10. Performance Optimization

### 10.1 Caching Strategy

```python
# Multi-Layer Caching
class CacheManager:
    def __init__(self, redis_client, local_cache_size=1000):
        self.redis = redis_client
        self.local_cache = LRUCache(maxsize=local_cache_size)
        
    async def get(self, key: str, fetch_fn: Callable, ttl: int = 3600):
        """
        Multi-layer cache lookup:
        1. Check local in-memory cache (fastest)
        2. Check Redis (fast)
        3. Fetch from source and cache (slow)
        """
        # Layer 1: Local cache
        if key in self.local_cache:
            return self.local_cache[key]
        
        # Layer 2: Redis cache
        cached = await self.redis.get(key)
        if cached:
            value = json.loads(cached)
            self.local_cache[key] = value
            return value
        
        # Layer 3: Fetch from source
        value = await fetch_fn()
        
        # Store in both caches
        await self.redis.setex(key, ttl, json.dumps(value))
        self.local_cache[key] = value
        
        return value
    
    async def invalidate(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        # Clear local cache
        keys_to_delete = [k for k in self.local_cache.keys() if fnmatch(k, pattern)]
        for key in keys_to_delete:
            del self.local_cache[key]
        
        # Clear Redis cache
        cursor = 0
        while True:
            cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
            if keys:
                await self.redis.delete(*keys)
            if cursor == 0:
                break

# Cache Warming
class CacheWarmer:
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
    
    async def warm_popular_searches(self):
        """Pre-populate cache with popular searches"""
        popular_queries = await self.get_popular_queries(limit=100)
        
        for query in popular_queries:
            try:
                # Generate embedding and cache
                embedding = await self.generate_embedding(query)
                await self.cache.set(
                    f"embedding:{hash(query)}",
                    embedding,
                    ttl=86400  # 24 hours
                )
            except Exception as e:
                logger.error("cache_warming_failed", query=query, error=str(e))
    
    async def warm_repository_metadata(self):
        """Pre-populate cache with repository metadata"""
        repos = await self.get_active_repositories()
        
        for repo in repos:
            metadata = await self.fetch_repo_metadata(repo.id)
            await self.cache.set(
                f"repo:{repo.id}:metadata",
                metadata,
                ttl=300  # 5 minutes
            )
```

### 10.2 Database Optimization

```python
# Query Optimization
class OptimizedQueries:
    @staticmethod
    def get_repository_with_stats(repo_id: str):
        """Optimized query with joins and aggregations"""
        return db.query(
            Repository,
            func.count(CodeFile.id).label('file_count'),
            func.sum(CodeFile.lines_of_code).label('total_lines'),
            func.count(TechDebtItem.id).label('debt_count')
        ).outerjoin(
            CodeFile, Repository.id == CodeFile.repository_id
        ).outerjoin(
            TechDebtItem, Repository.id == TechDebtItem.repository_id
        ).filter(
            Repository.id == repo_id
        ).group_by(
            Repository.id
        ).first()
    
    @staticmethod
    def search_with_pagination(query: str, page: int, limit: int):
        """Efficient pagination using keyset pagination"""
        # Use cursor-based pagination for better performance
        return db.query(CodeFile).filter(
            CodeFile.file_path.ilike(f"%{query}%")
        ).order_by(
            CodeFile.id
        ).limit(limit).offset((page - 1) * limit).all()

# Connection Pooling
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False
)

# Read Replicas for Scaling
class DatabaseRouter:
    def __init__(self, primary_url: str, replica_urls: List[str]):
        self.primary = create_engine(primary_url)
        self.replicas = [create_engine(url) for url in replica_urls]
        self.replica_index = 0
    
    def get_engine(self, write: bool = False):
        """Route queries to appropriate database"""
        if write:
            return self.primary
        
        # Round-robin load balancing for reads
        engine = self.replicas[self.replica_index]
        self.replica_index = (self.replica_index + 1) % len(self.replicas)
        return engine

# Batch Operations
class BatchProcessor:
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
    
    async def batch_insert_embeddings(self, embeddings: List[Tuple[str, np.ndarray]]):
        """Insert embeddings in batches for better performance"""
        for i in range(0, len(embeddings), self.batch_size):
            batch = embeddings[i:i + self.batch_size]
            
            # Prepare batch data
            ids = [item[0] for item in batch]
            vectors = [item[1].tolist() for item in batch]
            
            # Batch insert to ChromaDB
            self.vector_db.add(
                ids=ids,
                embeddings=vectors
            )
            
            await asyncio.sleep(0.1)  # Prevent overwhelming the database
```

### 10.3 Async Processing

```python
# Celery Task Queue Configuration
from celery import Celery

celery_app = Celery(
    'codearch',
    broker='amqp://guest:guest@rabbitmq:5672',
    backend='redis://redis:6379/0'
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour
    task_soft_time_limit=3300,  # 55 minutes
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=1000,
)

# Background Tasks
@celery_app.task(bind=True, max_retries=3)
def scan_repository(self, repo_id: str):
    """Scan repository in background"""
    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={'progress': 0})
        
        # Clone repository
        repo_path = clone_repository(repo_id)
        self.update_state(state='PROGRESS', meta={'progress': 10})
        
        # Scan files
        files = scan_files(repo_path)
        self.update_state(state='PROGRESS', meta={'progress': 30})
        
        # Parse code
        for i, file in enumerate(files):
            parse_code_file(file)
            progress = 30 + (i / len(files)) * 40
            self.update_state(state='PROGRESS', meta={'progress': progress})
        
        # Generate embeddings
        generate_embeddings(repo_id)
        self.update_state(state='PROGRESS', meta={'progress': 80})
        
        # Generate documentation
        generate_documentation(repo_id)
        self.update_state(state='PROGRESS', meta={'progress': 100})
        
        return {'status': 'completed', 'repo_id': repo_id}
        
    except Exception as e:
        logger.error("scan_failed", repo_id=repo_id, error=str(e))
        self.retry(exc=e, countdown=60 * (2 ** self.request.retries))

@celery_app.task
def generate_embeddings_batch(code_ids: List[str]):
    """Generate embeddings for multiple code snippets"""
    codes = fetch_codes(code_ids)
    embeddings = model.batch_generate(codes)
    store_embeddings(code_ids, embeddings)

@celery_app.task
def update_stale_documentation(repo_id: str):
    """Update stale documentation"""
    stale_docs = find_stale_docs(repo_id)
    for doc in stale_docs:
        regenerate_documentation(doc.id)

# Task Scheduling
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    'check-stale-docs-daily': {
        'task': 'tasks.check_stale_documentation',
        'schedule': crontab(hour=2, minute=0),  # 2 AM daily
    },
    'warm-cache-hourly': {
        'task': 'tasks.warm_cache',
        'schedule': crontab(minute=0),  # Every hour
    },
    'cleanup-old-data-weekly': {
        'task': 'tasks.cleanup_old_data',
        'schedule': crontab(day_of_week=0, hour=3, minute=0),  # Sunday 3 AM
    },
}
```

### 10.4 Vector Search Optimization

```python
# Optimized Vector Search
class OptimizedVectorSearch:
    def __init__(self, chroma_client):
        self.client = chroma_client
        self.collection = self.client.get_or_create_collection(
            name="code_embeddings",
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,  # Higher = better recall, slower indexing
                "hnsw:M": 16,  # Higher = better recall, more memory
                "hnsw:search_ef": 100  # Higher = better recall, slower search
            }
        )
    
    async def search_with_filters(
        self,
        query_embedding: np.ndarray,
        filters: Dict,
        top_k: int = 10
    ) -> List[SearchResult]:
        """Optimized search with pre-filtering"""
        # Build where clause for pre-filtering
        where_clause = {}
        if filters.get('language'):
            where_clause['language'] = filters['language']
        if filters.get('repository_id'):
            where_clause['repository_id'] = filters['repository_id']
        
        # Perform search with filters
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_clause if where_clause else None
        )
        
        return self.format_results(results)
    
    async def batch_search(
        self,
        query_embeddings: List[np.ndarray],
        top_k: int = 10
    ) -> List[List[SearchResult]]:
        """Batch search for better throughput"""
        results = self.collection.query(
            query_embeddings=[emb.tolist() for emb in query_embeddings],
            n_results=top_k
        )
        
        return [self.format_results(r) for r in results]
    
    def create_index_with_partitioning(self):
        """Partition embeddings by language for faster search"""
        languages = ['python', 'javascript', 'java', 'cpp', 'go']
        
        for lang in languages:
            collection_name = f"code_embeddings_{lang}"
            self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

# Approximate Nearest Neighbor (ANN) with FAISS
import faiss

class FAISSVectorSearch:
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        # Use IVF (Inverted File) index for faster search
        self.quantizer = faiss.IndexFlatL2(dimension)
        self.index = faiss.IndexIVFFlat(
            self.quantizer,
            dimension,
            100,  # Number of clusters
            faiss.METRIC_L2
        )
        self.is_trained = False
    
    def train(self, embeddings: np.ndarray):
        """Train the index on embeddings"""
        if not self.is_trained:
            self.index.train(embeddings)
            self.is_trained = True
    
    def add(self, embeddings: np.ndarray):
        """Add embeddings to index"""
        if not self.is_trained:
            self.train(embeddings)
        self.index.add(embeddings)
    
    def search(self, query_embedding: np.ndarray, k: int = 10):
        """Search for k nearest neighbors"""
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1),
            k
        )
        return indices[0], distances[0]
```

### 10.5 Frontend Optimization

```typescript
// Code Splitting and Lazy Loading
import { lazy, Suspense } from 'react';

const SearchPage = lazy(() => import('./pages/Search'));
const RepositoryPage = lazy(() => import('./pages/Repository'));
const DocumentationPage = lazy(() => import('./pages/Documentation'));

function App() {
  return (
    <Suspense fallback={<LoadingSpinner />}>
      <Routes>
        <Route path="/search" element={<SearchPage />} />
        <Route path="/repository/:id" element={<RepositoryPage />} />
        <Route path="/docs/:id" element={<DocumentationPage />} />
      </Routes>
    </Suspense>
  );
}

// React Query for Data Fetching and Caching
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

function useSearch(query: string) {
  return useQuery({
    queryKey: ['search', query],
    queryFn: () => api.search(query),
    staleTime: 5 * 60 * 1000, // 5 minutes
    cacheTime: 10 * 60 * 1000, // 10 minutes
    enabled: query.length > 0,
  });
}

// Virtual Scrolling for Large Lists
import { FixedSizeList } from 'react-window';

function SearchResults({ results }: { results: SearchResult[] }) {
  const Row = ({ index, style }: { index: number; style: React.CSSProperties }) => (
    <div style={style}>
      <SearchResultItem result={results[index]} />
    </div>
  );

  return (
    <FixedSizeList
      height={600}
      itemCount={results.length}
      itemSize={100}
      width="100%"
    >
      {Row}
    </FixedSizeList>
  );
}

// Debounced Search Input
import { useDebouncedValue } from '@mantine/hooks';

function SearchBar() {
  const [query, setQuery] = useState('');
  const [debouncedQuery] = useDebouncedValue(query, 300);
  const { data } = useSearch(debouncedQuery);

  return (
    <input
      value={query}
      onChange={(e) => setQuery(e.target.value)}
      placeholder="Search code..."
    />
  );
}

// Memoization for Expensive Computations
import { useMemo } from 'react';

function DependencyGraph({ data }: { data: GraphData }) {
  const processedData = useMemo(() => {
    // Expensive graph processing
    return processGraphData(data);
  }, [data]);

  return <D3Graph data={processedData} />;
}
```


## 11. Testing Strategy

### 11.1 Unit Testing

```python
# Test Structure
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Test Fixtures
@pytest.fixture
def sample_code():
    return """
    def validate_email(email: str) -> bool:
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    """

@pytest.fixture
def mock_vector_db():
    db = Mock()
    db.query.return_value = {
        'documents': [['sample code']],
        'metadatas': [[{'file_path': 'test.py', 'language': 'python'}]],
        'distances': [[0.1]]
    }
    return db

@pytest.fixture
def embedding_generator():
    return EmbeddingGenerator(model_name="microsoft/graphcodebert-base")

# Service Tests
class TestSearchService:
    def test_search_returns_results(self, mock_vector_db):
        service = SearchService(vector_db=mock_vector_db)
        results = service.search("validate email", top_k=10)
        
        assert len(results) > 0
        assert results[0].similarity_score > 0
        mock_vector_db.query.assert_called_once()
    
    def test_search_with_filters(self, mock_vector_db):
        service = SearchService(vector_db=mock_vector_db)
        results = service.search(
            "validate email",
            filters={'language': 'python'}
        )
        
        assert all(r.language == 'python' for r in results)
    
    def test_search_empty_query_raises_error(self, mock_vector_db):
        service = SearchService(vector_db=mock_vector_db)
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            service.search("")

class TestEmbeddingGenerator:
    def test_generate_embedding_shape(self, embedding_generator, sample_code):
        embedding = embedding_generator.generate_embedding(sample_code, "python")
        
        assert embedding.shape == (768,)
        assert embedding.dtype == np.float32
    
    def test_batch_generate_embeddings(self, embedding_generator):
        codes = ["def foo(): pass", "def bar(): pass", "def baz(): pass"]
        embeddings = embedding_generator.batch_generate(codes)
        
        assert embeddings.shape == (3, 768)
    
    @patch('transformers.AutoModel.from_pretrained')
    def test_model_loading_error_handling(self, mock_model):
        mock_model.side_effect = Exception("Model not found")
        
        with pytest.raises(Exception, match="Model not found"):
            EmbeddingGenerator(model_name="invalid/model")

class TestCodeParser:
    def test_parse_python_code(self, sample_code):
        parser = CodeParser()
        ast = parser.parse_to_ast(sample_code, Language.PYTHON)
        
        assert ast is not None
        functions = parser.extract_functions(ast)
        assert len(functions) == 1
        assert functions[0].name == "validate_email"
    
    def test_extract_imports(self):
        code = "import re\nfrom typing import List"
        parser = CodeParser()
        ast = parser.parse_to_ast(code, Language.PYTHON)
        imports = parser.extract_imports(ast)
        
        assert len(imports) == 2
        assert any(imp.module == "re" for imp in imports)
    
    def test_parse_invalid_syntax_raises_error(self):
        parser = CodeParser()
        invalid_code = "def foo( pass"
        
        with pytest.raises(SyntaxError):
            parser.parse_to_ast(invalid_code, Language.PYTHON)

# ML Model Tests
class TestContrastiveLearner:
    def test_nt_xent_loss_calculation(self):
        learner = ContrastiveLearner()
        z_i = torch.randn(32, 256)
        z_j = torch.randn(32, 256)
        
        loss = learner.nt_xent_loss(z_i, z_j, batch_size=32)
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_create_positive_pairs(self):
        learner = ContrastiveLearner()
        code = "def foo(): return 42"
        
        pairs = learner.create_positive_pairs(code)
        
        assert len(pairs) > 0
        assert all(isinstance(pair, tuple) for pair in pairs)
```

### 11.2 Integration Testing

```python
# API Integration Tests
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Test Database Setup
TEST_DATABASE_URL = "postgresql://test:test@localhost:5432/codearch_test"

@pytest.fixture(scope="function")
def test_db():
    engine = create_engine(TEST_DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    
    TestingSessionLocal = sessionmaker(bind=engine)
    db = TestingSessionLocal()
    
    yield db
    
    db.close()
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def client(test_db):
    def override_get_db():
        try:
            yield test_db
        finally:
            test_db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)

@pytest.fixture
def auth_headers():
    token = create_test_token(user_id="test_user", role="developer")
    return {"Authorization": f"Bearer {token}"}

# API Tests
class TestRepositoryAPI:
    def test_create_repository(self, client, auth_headers):
        response = client.post(
            "/api/v1/repositories",
            json={
                "git_url": "https://github.com/test/repo.git",
                "branch": "main"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["git_url"] == "https://github.com/test/repo.git"
        assert "id" in data
    
    def test_list_repositories(self, client, auth_headers, test_db):
        # Create test repositories
        create_test_repository(test_db, name="repo1")
        create_test_repository(test_db, name="repo2")
        
        response = client.get("/api/v1/repositories", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["data"]) == 2
    
    def test_unauthorized_access(self, client):
        response = client.get("/api/v1/repositories")
        assert response.status_code == 401

class TestSearchAPI:
    def test_search_code(self, client, auth_headers, test_db):
        # Setup test data
        setup_test_embeddings(test_db)
        
        response = client.get(
            "/api/v1/search",
            params={"q": "validate email", "limit": 10},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["query_time_ms"] < 2000  # Performance requirement
    
    def test_search_with_filters(self, client, auth_headers):
        response = client.get(
            "/api/v1/search",
            params={
                "q": "authentication",
                "language": "python",
                "limit": 5
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        results = response.json()["results"]
        assert all(r["language"] == "python" for r in results)

# Database Integration Tests
class TestDatabaseOperations:
    def test_repository_cascade_delete(self, test_db):
        # Create repository with files
        repo = Repository(name="test", git_url="https://test.git")
        test_db.add(repo)
        test_db.commit()
        
        file = CodeFile(repository_id=repo.id, file_path="test.py")
        test_db.add(file)
        test_db.commit()
        
        # Delete repository
        test_db.delete(repo)
        test_db.commit()
        
        # Verify cascade delete
        assert test_db.query(CodeFile).filter_by(repository_id=repo.id).count() == 0
    
    def test_knowledge_item_linking(self, test_db):
        repo = create_test_repository(test_db)
        file = create_test_file(test_db, repository_id=repo.id)
        
        knowledge = KnowledgeItem(
            repository_id=repo.id,
            title="Test Knowledge",
            content="Test content",
            linked_files=[{"file_id": str(file.id), "file_path": file.file_path}]
        )
        test_db.add(knowledge)
        test_db.commit()
        
        retrieved = test_db.query(KnowledgeItem).filter_by(id=knowledge.id).first()
        assert len(retrieved.linked_files) == 1
        assert retrieved.linked_files[0]["file_id"] == str(file.id)
```

### 11.3 End-to-End Testing

```python
# E2E Tests using Playwright
import pytest
from playwright.sync_api import Page, expect

@pytest.fixture(scope="session")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()

@pytest.fixture
def page(browser):
    context = browser.new_context()
    page = context.new_page()
    yield page
    context.close()

class TestUserFlows:
    def test_complete_onboarding_flow(self, page: Page):
        # Login
        page.goto("http://localhost:3000/login")
        page.fill('input[name="email"]', "test@example.com")
        page.fill('input[name="password"]', "password123")
        page.click('button[type="submit"]')
        
        # Wait for dashboard
        expect(page).to_have_url("http://localhost:3000/dashboard")
        
        # Add repository
        page.click('button:has-text("Add Repository")')
        page.fill('input[name="git_url"]', "https://github.com/test/repo.git")
        page.click('button:has-text("Add")')
        
        # Wait for scan to complete
        expect(page.locator('text=Scan completed')).to_be_visible(timeout=60000)
        
        # Navigate to repository
        page.click('text=test/repo')
        expect(page).to_have_url(re.compile(r".*/repository/.*"))
        
        # Verify repository details
        expect(page.locator('h1')).to_contain_text('test/repo')
    
    def test_search_flow(self, page: Page):
        page.goto("http://localhost:3000/search")
        
        # Enter search query
        page.fill('input[placeholder="Search code..."]', "validate email")
        
        # Wait for results
        expect(page.locator('.search-result')).to_have_count(10, timeout=5000)
        
        # Click on first result
        page.click('.search-result:first-child')
        
        # Verify code viewer opened
        expect(page.locator('.code-viewer')).to_be_visible()
    
    def test_ai_interview_flow(self, page: Page):
        page.goto("http://localhost:3000/knowledge")
        
        # Start interview
        page.click('button:has-text("Start Interview")')
        
        # Wait for first question
        expect(page.locator('.interview-question')).to_be_visible()
        
        # Answer question
        page.fill('textarea[name="answer"]', "This function validates email addresses using regex")
        page.click('button:has-text("Submit")')
        
        # Wait for next question or completion
        page.wait_for_timeout(2000)
        
        # End interview
        page.click('button:has-text("End Interview")')
        
        # Verify knowledge items created
        expect(page.locator('.knowledge-item')).to_have_count_greater_than(0)
```

### 11.4 Performance Testing

```python
# Load Testing with Locust
from locust import HttpUser, task, between

class CodeArchaeologistUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        # Login
        response = self.client.post("/api/v1/auth/login", json={
            "email": "test@example.com",
            "password": "password123"
        })
        self.token = response.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    @task(3)
    def search_code(self):
        queries = [
            "validate email",
            "authentication function",
            "database connection",
            "error handling"
        ]
        query = random.choice(queries)
        
        with self.client.get(
            "/api/v1/search",
            params={"q": query, "limit": 10},
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.elapsed.total_seconds() > 2:
                response.failure(f"Search took {response.elapsed.total_seconds()}s")
            elif response.status_code == 200:
                response.success()
    
    @task(1)
    def list_repositories(self):
        self.client.get("/api/v1/repositories", headers=self.headers)
    
    @task(2)
    def get_repository_details(self):
        # Assume we have repository IDs
        repo_id = random.choice(self.repo_ids)
        self.client.get(f"/api/v1/repositories/{repo_id}", headers=self.headers)
    
    @task(1)
    def get_tech_debt(self):
        repo_id = random.choice(self.repo_ids)
        self.client.get(f"/api/v1/debt/{repo_id}", headers=self.headers)

# Run: locust -f load_test.py --host=http://localhost:8000
```

### 11.5 Security Testing

```python
# Security Tests
class TestSecurity:
    def test_sql_injection_prevention(self, client, auth_headers):
        # Attempt SQL injection
        malicious_query = "'; DROP TABLE repositories; --"
        
        response = client.get(
            "/api/v1/search",
            params={"q": malicious_query},
            headers=auth_headers
        )
        
        # Should not cause error, query should be sanitized
        assert response.status_code in [200, 400]
        
        # Verify tables still exist
        response = client.get("/api/v1/repositories", headers=auth_headers)
        assert response.status_code == 200
    
    def test_xss_prevention(self, client, auth_headers):
        # Attempt XSS
        malicious_content = "<script>alert('XSS')</script>"
        
        response = client.post(
            "/api/v1/knowledge/items",
            json={
                "title": malicious_content,
                "content": "Test",
                "category": "architecture"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 201
        
        # Retrieve and verify content is escaped
        item_id = response.json()["id"]
        response = client.get(f"/api/v1/knowledge/items/{item_id}", headers=auth_headers)
        
        # Content should be escaped or sanitized
        assert "<script>" not in response.text
    
    def test_rate_limiting(self, client, auth_headers):
        # Make many requests quickly
        responses = []
        for _ in range(150):
            response = client.get("/api/v1/search", params={"q": "test"}, headers=auth_headers)
            responses.append(response.status_code)
        
        # Should get rate limited
        assert 429 in responses
    
    def test_unauthorized_access(self, client):
        endpoints = [
            "/api/v1/repositories",
            "/api/v1/search",
            "/api/v1/knowledge/items",
            "/api/v1/debt/test-repo-id"
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 401
    
    def test_secret_detection(self):
        scanner = SecretScanner()
        
        code_with_secret = '''
        AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
        AWS_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        '''
        
        matches = scanner.scan_code(code_with_secret, "config.py")
        
        assert len(matches) >= 2
        assert any(m.type == "aws_access_key" for m in matches)
        assert any(m.type == "aws_secret_key" for m in matches)
```


## 12. Deployment & Release Strategy

### 12.1 CI/CD Pipeline

```yaml
# GitHub Actions Workflow
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          docker-compose -f docker-compose.test.yml up --abort-on-container-exit
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build images
        run: docker-compose build
      - name: Push to registry
        run: |
          docker tag codearch/api-gateway:latest registry.io/codearch/api-gateway:${{ github.sha }}
          docker push registry.io/codearch/api-gateway:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/api-gateway api-gateway=registry.io/codearch/api-gateway:${{ github.sha }}
          kubectl rollout status deployment/api-gateway
```

### 12.2 Release Phases

**Phase 1: MVP (Months 1-3)**
- Core ingestion and parsing
- Basic semantic search
- Simple documentation generation
- PostgreSQL + ChromaDB only

**Phase 2: Enhanced Features (Months 4-6)**
- SimCLR contrastive learning
- AI interview system
- Technical debt tracking
- Neo4j knowledge graph

**Phase 3: Production Ready (Months 7-9)**
- Security scanning
- IDE plugins
- Advanced analytics
- Full monitoring stack

**Phase 4: Scale & Optimize (Months 10-12)**
- Performance optimization
- Multi-tenant support
- Enterprise features
- SaaS deployment

## 13. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| ML model accuracy | Extensive testing, human-in-the-loop validation, fallback to simpler models |
| Scalability issues | Horizontal scaling, caching, async processing, load testing |
| Data privacy | On-premise deployment, encryption, compliance certifications |
| Integration complexity | Start with GitHub only, phased rollout, comprehensive documentation |
| User adoption | Beta program, training materials, gradual feature rollout |

## 14. Success Metrics

**Technical KPIs:**
- Search latency: <2s (p95)
- System uptime: >99.5%
- Code coverage: >80%
- API response time: <500ms (p95)

**Business KPIs:**
- Onboarding time reduction: 50%
- Developer productivity increase: 20%
- Bug reduction: 30%
- User adoption: 80% within 6 months

## 15. Future Enhancements

- Real-time collaboration features
- AI-powered code review
- Automated refactoring suggestions
- Integration with CI/CD pipelines
- Mobile applications
- Video documentation generation
- Multi-repository analysis
- Custom ML model training on customer data

## 16. Conclusion

CodeArchaeologist is designed as a scalable, production-ready platform that combines modern ML techniques with traditional software engineering practices. The architecture supports:

- **Scalability**: Microservices, horizontal scaling, distributed processing
- **Performance**: Multi-layer caching, async processing, optimized queries
- **Security**: Encryption, RBAC, secret detection, compliance
- **Maintainability**: Clean architecture, comprehensive testing, monitoring
- **Extensibility**: Plugin architecture, API-first design, modular components

The system is built to handle repositories up to 1M LOC while maintaining sub-2-second search response times and providing intelligent insights that reduce onboarding time by 50% and increase developer productivity by 20%.

Key architectural decisions:
- **GraphCodeBERT + SimCLR** for semantic understanding
- **Multi-database approach** (PostgreSQL, ChromaDB, Neo4j, Redis) for optimal data storage
- **Microservices architecture** for independent scaling and deployment
- **Async processing** with Celery for long-running tasks
- **Comprehensive observability** with Prometheus, Grafana, and distributed tracing

This design provides a solid foundation for building an AI-powered code documentation and knowledge management platform that addresses real developer pain points while maintaining enterprise-grade reliability and security.
