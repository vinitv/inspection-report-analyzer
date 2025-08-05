## Property Inspection Report Analyzer for Southern California First-Time Home Buyers

### ARTIFACTS

- [Notebook](./Complete_Inspection_RAG_System.ipynb)
- [Home Repair Cost API](https://repair-cost-api-618596951812.us-central1.run.app/docs) Custom Built/Hosted on Cloud Run
- [Loom](https://www.loom.com/share/ab09b4917fba4ce9ac551828fda110ed?sid=b53033a5-9b5a-4eb8-9fa9-f523a7f6c9ce)

### Challenge Tasks


### Task 1: Articulate the problem and the user of your application

First-time home buyers in Southern California receive overwhelming 50-200 page inspection reports that they cannot interpret, leading to poor purchase decisions and unexpected repair costs.


#### Why This is a Problem

##### ✅ Answer:
First-time home buyers typically lack the technical knowledge to understand inspection reports, which are written by professionals for professionals. These reports contain critical information about structural issues, electrical problems, plumbing concerns, and HVAC systems, but the language is technical and the financial implications are unclear. Without proper interpretation, buyers either walk away from good deals due to minor issues or proceed with purchases that will cost them thousands in unexpected repairs.
The problem is particularly acute in Southern California's competitive market where buyers often have limited time to make decisions and may waive inspection contingencies. When they do get inspections, the reports become sources of anxiety rather than helpful decision-making tools. Many buyers rely on friends or family for advice, leading to inconsistent and potentially costly guidance.

Potential User Questions

- How much will it cost to fix the issues found in this inspection?
- Which problems need immediate attention versus long-term planning?
- Is this electrical issue a safety concern or cosmetic?
- Should I negotiate the price based on these findings?
- What are the typical costs for HVAC repairs in my area?
- Are these foundation concerns serious or normal settling?


### Task 2: Proposed Solution

##### ✅ Answer:
The Property Inspection Report Analyzer will transform complex inspection documents into clear, actionable insights with specific cost estimates and prioritized repair timelines. Users will upload their inspection PDF and receive an easy to understand breakdown of issues categorized by urgency (immediate safety concerns, items to address within 6 months, and long-term maintenance), complete with Southern California-specific repair cost estimates and other recommendations.


2. Describe the tools you plan to use in each part of your stack.  Write one sentence on why you made each tooling choice.

##### ✅ Answer:

1. LLM: GPT-4o-mini (in prod app and used gpt-4o in notebook)
I chose GPT-4o-mini because it provides excellent document understanding for complex inspection reports with technical terminology. It's cost-effective for real-time chat interactions while maintaining high performance.

2. Embedding Model: OpenAI text-embedding-3-small
I selected text-embedding-3-small because it effectively handles technical construction terminology and maintains semantic relationships. It's optimized for speed and cost efficiency in vector similarity search.

3. Orchestration: LangGraph
I implemented LangGraph because it provides sophisticated multi-agent coordination with intelligent routing strategies (TOOL_SEARCH, RAG_SIMPLE, RAG_COMPLEX). It enables flexible state management for complex inspection analysis workflows.

4. Vector Database: Qdrant (In-Memory)
I chose Qdrant in-memory mode because it provides high-performance vector similarity search with excellent LangChain integration. It enables fast retrieval without requiring external infrastructure setup.

5. Monitoring: LangSmith + Python Logging
I implemented LangSmith tracing with comprehensive Python logging because it provides detailed visibility into agent decision-making and tool usage. It enables debugging and performance optimization of the complex agent workflows.

6. Evaluation: RAGAS Framework
I selected RAGAS because it's the standard framework for RAG system evaluation with comprehensive metrics. 

7. User Interface: HTML

8. Serving & Inference: FastAPI with Uvicorn
I chose FastAPI with Uvicorn because it provides high-performance async API serving with automatic OpenAPI documentation. It seamlessly integrates with both frontend interfaces and handles concurrent requests efficiently.

----

3. Where will you use an agent or agents?  What will you use “agentic reasoning” for in your app?
##### ✅ Answer:

The application uses **one sophisticated agent** with **agentic reasoning** for intelligent decision-making and task routing. Here's how:

## **1. Intelligent Question Analysis & Strategy Selection**
The agent employs agentic reasoning to analyze incoming questions and automatically determine the best approach:

- **Simple Cost Questions** → Routes to forced tool usage
- **Complex Budget/Prioritization Questions** → Routes to RAG analysis + cost tools  
- **Inspection Findings Questions** → Routes to document analysis
- **General Questions** → Routes to web search tools

## **2. Multi-Step Reasoning for Complex Queries**
For complex budget questions like *"What should I prioritize with $5,000?"*, the agent uses sophisticated reasoning:

1. **Document Analysis**: First analyzes the inspection report to extract all findings
2. **Issue Categorization**: Identifies repair types (HVAC, electrical, plumbing, etc.)
3. **Cost Integration**: Calls repair cost tools for each identified issue
4. **Priority Reasoning**: Combines safety urgency, cost estimates, and budget constraints
5. **Recommendation Synthesis**: Creates prioritized action plans

## **3. Adaptive Tool Selection**
The agent makes intelligent decisions about when and how to use tools:

- **Forced Tool Usage**: Automatically triggers cost estimation tools when LLM fails to call them
- **Conditional Routing**: Uses different strategies (TOOL_SEARCH, RAG_SIMPLE, RAG_COMPLEX) based on question complexity
- **Fallback Mechanisms**: Falls back to web search when specialized tools fail

## **4. Context-Aware Reasoning**
The agent maintains context awareness throughout conversations:

- **Session Memory**: Remembers previous analysis and inspection context
- **Enhanced Questioning**: Automatically adds inspection report context to follow-up questions
- **Strategy Adaptation**: Adjusts approach based on available information

## **What Agentic Reasoning Accomplishes**

1. **Transforms Technical Jargon**: Converts complex inspection terminology into homeowner-friendly explanations
2. **Financial Decision Support**: Provides California-specific cost estimates with priority rankings
3. **Safety Risk Assessment**: Distinguishes between immediate safety concerns and cosmetic issues
4. **Budget Optimization**: Creates actionable spending plans based on actual inspection findings

The agentic system essentially acts as an **intelligent intermediary** that understands both the technical complexity of inspection reports and the practical needs of first-time home buyers, using reasoning to bridge that gap effectively.


### Task 3: Dealing with the Data


1. Describe all of your data sources and external APIs, and describe what you’ll use them for.

##### ✅ Answer:

### **Primary Data Sources:**
- **Inspection Reports PDF Dataset**: Located in `data/inspection-reports/` and `data/my-report/` - serves as the knowledge base for RAG system containing sample property inspection reports
- **User-Uploaded PDF Reports**: Real-time inspection reports uploaded by users via the web interface for analysis

### **External APIs Used:**

1. **OpenAI API** - Powers the GPT-4o-mini LLM and text-embedding-3-small model for document understanding and semantic search
2. **Repair Cost Estimation API** - Custom API at `https://repair-cost-api-618596951812.us-central1.run.app` providing Southern California-specific repair cost estimates
3. **Tavily Search API** - Web search tool for fallback when specialized repair cost data isn't available
4. **Cohere API** - Used for reranking and compression of search results (optional enhancement)
5. **LangSmith API** - Monitoring and tracing of agent performance and conversation flows


2. Describe the default chunking strategy that you will use.  Why did you make this decision?

##### ✅ Answer:

### **Strategy: Recursive Character Text Splitter**
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Method**: `RecursiveCharacterTextSplitter` from LangChain

### **Why This Decision:**
This chunking strategy was chosen because inspection reports contain structured technical information that needs to maintain context while being digestible for the LLM. The 1000-character chunks ensure complete inspection findings (like "electrical panel shows signs of overheating") stay together, while the 200-character overlap prevents critical information from being split across boundaries. This size is optimal for preserving the relationship between problems and their technical details, ensuring accurate cost estimation and priority assessment.


3. [Optional] Will you need specific data for any other part of your application?   If so, explain.

##### ✅ Answer:

### **Synthetic Evaluation Dataset:**
- **Purpose**: RAGAS evaluation framework requires test questions with known answers
- **Source**: Generated from evaluation documents in `data/inspection-reports/` 
- **Usage**: Validates system performance with metrics like Context Recall (0.619), Faithfulness (0.843), and Answer Relevancy (0.601)

### **Regional Pricing Data:**
- **Purpose**: Southern California-specific cost multipliers and repair estimates
- **Integration**: Built into the custom Repair Cost API to provide accurate local pricing rather than generic national averages
- **Importance**: Critical for the target audience of SoCal first-time home buyers who face 20-50% higher repair costs than national averages

### Task 4: Building a Quick End-to-End Agentic RAG Prototype

##### ✅ Answer: DONE. CHECK LOOM AND CODE HERE


### Task 5: Creating a Golden Test Data Set

1. Assess your pipeline using the RAGAS framework including key metrics faithfulness, response relevance, context precision, and context recall.  Provide a table of your output results.

##### ✅ Answer:

| Method | Overall | Context Recall | Faithfulness | Answer Relevancy |
|--------|---------|----------------|--------------|------------------|
| Baseline | 0.636 | 0.500 | 0.800 | 0.610 |

2. What conclusions can you draw about the performance and effectiveness of your pipeline with this information?

##### ✅ Answer:

The RAGAS evaluation results reveal a pipeline with mixed performance that's functional but needs improvement for the critical home-buying use case. The strong faithfulness score (0.800) is encouraging, meaning when the system retrieves relevant information, it provides accurate and reliable answers about costs and safety concerns - crucial for first-time buyers making high-stakes decisions. However, the concerning context recall score (0.500) indicates the system only finds half of the relevant inspection findings, which could lead to missed critical issues or incomplete repair recommendations. The moderate answer relevancy (0.610) suggests responses sometimes lack focus on the specific user question. Overall, while the pipeline demonstrates reliability when it finds the right information, the incomplete retrieval poses significant risks for comprehensive inspection analysis, making it suitable for basic guidance but requiring optimization before handling complex prioritization decisions that could impact major financial investments.



### Task 6: The Benefits of Advanced Retrieval

## Retrieval Techniques for Inspection Report Analysis

##### ✅ Answer:

### **1. Baseline (Naive) Vector Similarity Search**
This technique uses standard semantic similarity search with embeddings to find the most relevant inspection report sections. I believe this will be useful as a foundation because it provides straightforward retrieval of content with semantic similarity to user questions about specific inspection findings.

### **2. Multi-Query Retriever**
This technique uses the LLM to generate multiple query variations and combines results from all variations to improve recall. I believe this will be especially useful for inspection reports because users often ask questions in different ways (e.g., "electrical issues" vs "wiring problems") and this ensures comprehensive coverage of related technical terminology.

### **3. Parent-Document Retriever (Small-to-Big Strategy)**
This technique splits documents into small chunks for precise retrieval but returns larger parent sections for complete context. I believe this will be critical for inspection reports because individual issues (like "loose electrical connections") need surrounding context (safety implications, location details, recommended actions) to provide meaningful cost estimates and prioritization advice.

### **Why These Techniques Matter for First-Time Home Buyers:**
Each technique addresses a specific challenge: baseline provides speed, multi-query ensures nothing is missed despite varied terminology, and parent-document maintains the contextual relationships essential for accurate cost estimation and safety assessment.


### Task 7: Assessing Performance

##### ✅ Answer:

## RAGAS Evaluation Results

| Technique | Overall | Context Recall | Faithfulness | Answer Relevancy |
|-----------|---------|----------------|--------------|------------------|
| **Multi-Query** | **0.688** | **0.619** | **0.843** | 0.601 |
| Baseline | 0.636 | 0.500 | 0.800 | **0.610** |
| Parent-Document | 0.559 | 0.564 | 0.652 | 0.461 |

**Best Performer: Multi-Query (0.688)**

---

## 1. Performance Comparison Analysis

### **Multi-Query vs. Baseline Improvements:**
- **Overall Performance**: +8.2% improvement (0.688 vs 0.636)
- **Context Recall**: +23.8% improvement (0.619 vs 0.500) - **Most significant gain**
- **Faithfulness**: +5.4% improvement (0.843 vs 0.800)
- **Answer Relevancy**: -1.5% decrease (0.601 vs 0.610) - Minor trade-off

### **Key Insights:**
The **Multi-Query retriever significantly improves context recall**, which was the biggest weakness in the baseline system. This means users are now much less likely to miss critical inspection findings. The slight decrease in answer relevancy is acceptable given the substantial improvement in finding relevant information. The Parent-Document approach underperformed, likely because inspection reports need precise technical details rather than broad context sections.

---

## 2. Planned Improvements for Second Half of Course
- Make the agent even smarter with ability to analyze a property 
- Use Google ADK or OpenAI SDK