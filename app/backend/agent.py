"""
Enhanced Inspection Report Analyzer Agent
Updated with multi-query retriever and improved tools from notebook
"""

import os
import logging
from typing import Dict, List, Optional, Annotated
from datetime import datetime

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode

import requests
from typing import TypedDict, Literal
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

logger = logging.getLogger(__name__)

class EnhancedAgentState(TypedDict):
    """Enhanced state for our tool-enabled agentic RAG system"""
    messages: Annotated[list, add_messages]
    question: str
    context: str
    tool_calls: list
    strategy: str
    reasoning: str
    final_answer: str

class InspectionAnalyzer:
    """
    Enhanced inspection report analyzer with agentic capabilities
    """
    
    def __init__(self):
        self.is_ready = False
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.retrievers = {}
        self.agent = None
        self.tools = []
        
    async def initialize(self):
        """Initialize all components"""
        try:
            logger.info("🔄 Initializing Inspection Analyzer...")
            
            # Initialize LLM and embeddings
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            
            # Setup tools
            self._setup_tools()
            
            # Load inspection reports and create vector store
            await self._setup_vectorstore()
            
            # Setup retrievers
            self._setup_retrievers()
            
            # Build enhanced agent
            self._build_agent()
            
            self.is_ready = True
            logger.info("✅ Inspection Analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize analyzer: {str(e)}")
            raise
    
    def _setup_tools(self):
        """Setup enhanced tools from notebook"""
        
        # Dynamic repair type mapping using actual API data
        def get_api_repair_types():
            """Fetch actual repair types from API"""
            try:
                api_url = "https://repair-cost-api-618596951812.us-central1.run.app/api/v1/repair-types"
                headers = {"x-api-key": os.getenv("REPAIR_API_KEY", "demo-key-2024")}
                
                response = requests.get(api_url, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    return data.get('repair_types', [])
                else:
                    return []
            except Exception as e:
                return []

        def map_repair_type(repair_description: str) -> str:
            """Map user description to actual API repair type"""
            api_repair_types = get_api_repair_types()
            if not api_repair_types:
                # Fallback list if API is unavailable
                fallback_types = ["roof_repair", "electrical_panel_replacement", "hvac_repair", "plumbing_repair"]
                desc = repair_description.lower()
                if any(word in desc for word in ['roof', 'roofing', 'shingle']):
                    return "roof_repair"
                elif any(word in desc for word in ['electrical', 'panel', 'breaker']):
                    return "electrical_panel_replacement"
                elif any(word in desc for word in ['hvac', 'heating', 'ac', 'furnace']):
                    return "hvac_repair"
                elif any(word in desc for word in ['plumbing', 'leak', 'pipe', 'water']):
                    return "plumbing_repair"
                else:
                    return fallback_types[0]
            
            desc = repair_description.lower()
            
            # First try exact matches
            for repair in api_repair_types:
                repair_type = repair['repair_type']
                if repair_type.lower() in desc or desc.replace(' ', '_') in repair_type.lower():
                    return repair_type
            
            # Then try keyword matching
            for repair in api_repair_types:
                repair_type = repair['repair_type'].lower()
                repair_desc = repair.get('description', '').lower()
                
                # Check common keywords against repair type and description
                if any(word in desc for word in ['roof', 'shingle', 'leak']) and 'roof' in repair_type:
                    return repair['repair_type']
                elif any(word in desc for word in ['electrical', 'wiring', 'outlet', 'panel']) and 'electrical' in repair_type:
                    return repair['repair_type']
                elif any(word in desc for word in ['hvac', 'heating', 'ac', 'furnace', 'air condition']) and any(keyword in repair_type for keyword in ['hvac', 'heating', 'ac']):
                    return repair['repair_type']
                elif any(word in desc for word in ['plumbing', 'pipe', 'water', 'leak']) and 'plumbing' in repair_type:
                    return repair['repair_type']
                elif any(word in desc for word in ['paint', 'painting']) and 'paint' in repair_type:
                    return repair['repair_type']
                elif any(word in desc for word in ['flooring', 'floor', 'hardwood', 'carpet']) and any(keyword in repair_type for keyword in ['flooring', 'floor']):
                    return repair['repair_type']
            
            # Return first available repair type as fallback
            return api_repair_types[0]['repair_type']

        @tool
        def get_repair_cost_estimate(
            repair_type: Annotated[str, "Type of repair (e.g., 'roof repair', 'electrical panel', 'HVAC system')"],
            zip_code: Annotated[str, "ZIP code"] = "90210"
        ) -> str:
            """Get repair cost estimates using the Home Repair Cost API"""
            try:
                # Ensure zip_code is 5 digits
                if len(zip_code) != 5 or not zip_code.isdigit():
                    zip_code = "90210"  # Default to Beverly Hills
                
                mapped_repair_type = map_repair_type(repair_type)
                logger.info(f"🔍 Mapped '{repair_type}' to '{mapped_repair_type}'")
                
                api_url = f"https://repair-cost-api-618596951812.us-central1.run.app/api/v1/repair-cost/{mapped_repair_type}"
                logger.info(f"🌐 Making API call to: {api_url}")
                logger.info(f"📍 Using zip_code: {zip_code}")
                headers = {"x-api-key": os.getenv("REPAIR_API_KEY", "demo-key-2024")}
                params = {"zip_code": zip_code, "scope": "average"}
                
                response = requests.get(api_url, headers=headers, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    cost_est = data.get('cost_estimate', {})
                    location_info = data.get('location', {})
                    details = data.get('details', {})
                    
                    result = f"""💰 **{repair_type.title()} Cost Estimate**
📍 Location: {location_info.get('region', 'Unknown')} (Zip: {zip_code})
💲 Cost Range: ${cost_est.get('low', 0):,} - ${cost_est.get('high', 0):,}
📊 Average: ${cost_est.get('average', 0):,}"""
                    if details.get('description'):
                        result += f"\n📝 Details: {details['description']}"
                    if details.get('labor_hours'):
                        result += f"\n⏱️ Labor: {details['labor_hours']}"
                    
                    return result
                else:
                    error_msg = f"❌ API Error (Status {response.status_code})"
                    try:
                        error_detail = response.json()
                        error_msg += f": {error_detail}"
                    except:
                        error_msg += f": {response.text}"
                    logger.error(error_msg)
                    return f"Cost estimate unavailable for {repair_type}. Please try using get_available_repair_types to see valid options."
            except Exception as e:
                error_msg = f"Error getting cost estimate: {str(e)}"
                logger.error(f"⚠️ {error_msg}")
                return error_msg

        @tool
        def get_available_repair_types() -> str:
            """Get a list of all available repair types supported by the cost estimation API"""
            try:
                api_url = "https://repair-cost-api-618596951812.us-central1.run.app/api/v1/repair-types"
                headers = {"x-api-key": os.getenv("REPAIR_API_KEY", "demo-key-2024")}
                
                response = requests.get(api_url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    repair_types = data.get('repair_types', [])
                    categories = data.get('categories', [])
                    total_count = data.get('total_count', len(repair_types))
                    
                    result = f"Available Repair Types ({total_count} total across {len(categories)} categories):\n\n"
                    
                    # Group by category
                    for category in categories:
                        category_repairs = [r for r in repair_types if r['category'] == category]
                        if category_repairs:
                            result += f"**{category}** ({len(category_repairs)} types):\n"
                            for repair in category_repairs[:3]:  # Show first 3 per category
                                cost_range = repair.get('average_cost_range', 'N/A')
                                result += f"  • {repair['repair_type']} - {cost_range}\n"
                            if len(category_repairs) > 3:
                                result += f"  ... and {len(category_repairs) - 3} more\n"
                            result += "\n"
                    
                    return result
            else:
                    return "Unable to fetch repair types from API"
            except Exception as e:
                return f"Error fetching repair types: {e}"

        # Web search tool
        tavily_tool = TavilySearchResults(max_results=3)

        self.tools = [get_repair_cost_estimate, get_available_repair_types, tavily_tool]
        logger.info(f"Setup {len(self.tools)} enhanced tools for agent")
    
    async def _setup_vectorstore(self):
        """Setup vector store with inspection reports using Qdrant"""
        try:
            # Load inspection reports from data directory
            documents = []
            data_dir = "data/my-report"  # Use the primary report directory
            
            if os.path.exists(data_dir):
                for filename in os.listdir(data_dir):
                    if filename.lower().endswith('.pdf'):
                        filepath = os.path.join(data_dir, filename)
                        loader = PyMuPDFLoader(filepath)
                        docs = loader.load()
                        documents.extend(docs)
                        logger.info(f"Loaded {len(docs)} pages from {filename}")
            
            # Fallback to inspection-reports if my-report is empty
            if not documents:
                data_dir = "data/inspection-reports"
                if os.path.exists(data_dir):
                    for filename in os.listdir(data_dir):
                        if filename.lower().endswith('.pdf'):
                            filepath = os.path.join(data_dir, filename)
                            loader = PyMuPDFLoader(filepath)
                            docs = loader.load()
                            documents.extend(docs)
                            logger.info(f"Loaded {len(docs)} pages from {filename}")
            
            if not documents:
                logger.warning("No inspection reports found in data directories")
                # Create sample document for vectorstore initialization
                from langchain.schema import Document
                documents = [Document(
                    page_content="Sample California property inspection report content for demonstration purposes.",
                    metadata={"source": "sample"}
                )]
            
            # Split documents with improved settings
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            self.split_docs = text_splitter.split_documents(documents)
            
            # Setup Qdrant client and vector store
            self.client = QdrantClient(":memory:")
            sample_embedding = self.embeddings.embed_query("test")
            embedding_dim = len(sample_embedding)
            
            self.client.create_collection(
                collection_name="inspection_knowledge",
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
            )
            
            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name="inspection_knowledge", 
                embedding=self.embeddings,
            )
            
            # Add documents to vector store
            _ = self.vectorstore.add_documents(documents=self.split_docs)
            
            logger.info(f"Created vector store with {len(self.split_docs)} document chunks")
            
        except Exception as e:
            logger.error(f"Failed to setup vector store: {str(e)}")
            raise
    
    def _setup_retrievers(self):
        """Setup multi-query retriever as primary strategy"""
        try:
            # Baseline retriever
            baseline_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 8})
            
            # Multi-Query Retriever (primary strategy from notebook)
            self.multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=baseline_retriever,
                llm=self.llm
            )
            
            # Store retrievers
            self.retrievers = {
                'baseline': baseline_retriever,
                'multi_query': self.multi_query_retriever,
                'primary': self.multi_query_retriever  # Use multi-query as primary
            }
            
            logger.info("Setup multi-query retriever as primary strategy")
            
        except Exception as e:
            logger.error(f"Failed to setup retrievers: {str(e)}")
            # Fallback to basic retriever
            self.retrievers = {'baseline': self.vectorstore.as_retriever()}
    
    def _build_agent(self):
        """Build the enhanced agent with improved tool routing"""
        try:
            # Helper functions
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            # Enhanced tool prompt from notebook
            enhanced_tool_prompt = """You are an intelligent property inspection assistant with access to specialized tools. 

CRITICAL: For ANY cost-related question, you MUST use the repair cost tools.

INTELLIGENT TOOL USAGE STRATEGY:

For SIMPLE COST questions ("How much does X cost?", "cost to replace Y"):
  - ALWAYS call "get_repair_cost_estimate" with the repair type mentioned
  - If unsure about repair types, first call "get_available_repair_types"
  - Examples: "HVAC replacement", "electrical panel", "roof repair", "plumbing"

For COMPLEX BUDGET/PRIORITIZATION questions ("What should I prioritize with $X budget?"):
  1. ANALYZE the inspection document for findings and issues
  2. IDENTIFY specific repairs mentioned in the inspection 
  3. GET cost estimates for those specific repairs using repair tools
  4. PRIORITIZE based on safety, urgency, and budget constraints

For INSPECTION FINDINGS questions ("What issues were found?"):
  - Analyze the inspection document content directly

For GENERAL questions or when repair tools fail:
  - Use "tavily_tool" for web search

MANDATORY TOOL USAGE:
- Questions with "cost", "price", "how much", "estimate" → USE get_repair_cost_estimate
- Questions about "prioritize", "budget" → USE inspection analysis + cost tools
- Questions about "found", "issues", "problems" → USE document analysis

ALWAYS use tools when appropriate. Don't just provide general answers for cost questions."""

            # Enhanced LLM with routing instructions
            enhanced_tool_llm = self.llm.bind_tools(self.tools)
            tool_node = ToolNode(self.tools)
            
            # Simplified state definition for our agent
            class AgentState(TypedDict):
                messages: Annotated[list, add_messages]
                question: str

            def agent_node(state: AgentState):
                """Enhanced agent node with intelligent tool routing"""
                messages = state["messages"]
                
                # Add the routing instructions as system context
                if not messages or not isinstance(messages[0], HumanMessage):
                    system_context = f"{enhanced_tool_prompt}\n\nUser question: {state['question']}"
                    enhanced_messages = [HumanMessage(content=system_context)] + messages
                else:
                    enhanced_messages = messages
                
                response = enhanced_tool_llm.invoke(enhanced_messages)
                return {"messages": [response]}

            def should_continue(state: AgentState) -> Literal["tools", "rag_analysis", "force_tools", END]:
                """Smart conditional routing - handle both simple and complex questions"""
                last_message = state["messages"][-1]
                question = state.get("question", "")
                
                logger.info(f"🎯 ROUTING DECISION for: '{question}'")
                
                # PRIORITY 1: Check for tool calls first (LLM decided to use tools)
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    return "tools"
                
                # PRIORITY 2: Force tools for simple cost questions if LLM didn't call them
                cost_keywords = ['cost', 'price', 'estimate', 'expensive', 'cheap', 'money', 'dollar', '$']
                simple_cost_patterns = ['how much', 'what does', 'cost to', 'price of', 'estimate for', 'cost of', "what's the cost"]
                complex_budget_keywords = ['prioritize', 'priority', 'most important', 'urgent', 'critical', 'budget']
                
                has_cost_context = any(keyword in question.lower() for keyword in cost_keywords)
                has_simple_cost = any(pattern in question.lower() for pattern in simple_cost_patterns)
                has_prioritization = any(keyword in question.lower() for keyword in complex_budget_keywords)
                
                # Force tools for ANY cost question that doesn't involve prioritization
                if has_cost_context and not has_prioritization:
                    logger.info(f"🔧 ROUTING TO FORCE_TOOLS: cost_context={has_cost_context}, prioritization={has_prioritization}")
                    return "force_tools"
                
                # PRIORITY 3: Complex budget/prioritization questions need RAG + tools
                if has_cost_context and has_prioritization:
                    # Complex questions: "What should I prioritize with $5000?" - need inspection analysis
                    return "rag_analysis"
                
                # PRIORITY 4: Check for inspection findings questions
                inspection_keywords = ['found', 'issues', 'problems', 'condition', 'findings', 'defects', 'damage']
                if any(keyword in question.lower() for keyword in inspection_keywords):
                    logger.info(f"🏠 ROUTING TO RAG_ANALYSIS: inspection question detected")
                    return "rag_analysis"
                
                # PRIORITY 5: Everything else ends
                logger.info(f"🔚 ROUTING TO END: no special routing needed")
                return END

            def force_tools_node(state: AgentState):
                """Force tool usage for simple cost questions using proper mapping"""
                question = state["question"]
                
                logger.info(f"🔧 FORCE_TOOLS_NODE executing for: '{question}'")
                
                # Extract repair type from question using improved mapping
                # This will be properly mapped by the actual tool
                repair_type = "general repair"
                q_lower = question.lower()
                
                if any(word in q_lower for word in ["hvac", "heating", "cooling", "air condition", "furnace", "ac"]):
                    repair_type = "HVAC system"
                elif any(word in q_lower for word in ["electrical", "panel", "wiring", "outlet", "breaker"]):
                    repair_type = "electrical panel"
                elif any(word in q_lower for word in ["roof", "roofing", "shingle"]):
                    repair_type = "roof repair"
                elif any(word in q_lower for word in ["plumbing", "pipe", "water", "leak", "faucet"]):
                    repair_type = "plumbing repair"
                elif any(word in q_lower for word in ["flooring", "floor", "hardwood", "carpet"]):
                    repair_type = "flooring"
                elif any(word in q_lower for word in ["paint", "painting"]):
                    repair_type = "painting"
                
                logger.info(f"🔧 EXTRACTED REPAIR TYPE: '{repair_type}' from question")
                
                # Create proper tool call using the same structure as notebook
                from langchain_core.messages import AIMessage
                
                # Create tool call for cost estimate (matches notebook exactly)
                tool_call = {
                    "name": "get_repair_cost_estimate",
                    "args": {"repair_type": repair_type, "zip_code": "90210"},
                    "id": "forced_cost_call",
                    "type": "tool_call"
                }
                
                # Create AI message with tool call
                ai_message = AIMessage(
                    content=f"I'll get the cost estimate for {repair_type} replacement/repair using the repair cost API.",
                    tool_calls=[tool_call]
                )
                
                return {"messages": [ai_message]}

            def rag_analysis_node(state: AgentState):
                """Smart RAG analysis - handles both findings and complex budget questions"""
                question = state["question"]
                
                # Check if this is a complex budget/prioritization question
                complex_budget_keywords = ['prioritize', 'priority', 'most important', 'urgent', 'critical']
                cost_keywords = ['cost', 'price', 'estimate', 'budget', 'expensive', 'cheap', 'money', 'dollar', '$']
                
                has_budget_context = any(keyword in question.lower() for keyword in cost_keywords)
                has_prioritization = any(keyword in question.lower() for keyword in complex_budget_keywords)
                
                if has_budget_context and has_prioritization:
                    # Complex budget question - analyze inspection findings AND get cost estimates
                    logger.info(f"🏠💰 BUDGET QUESTION: Getting inspection findings + cost estimates")
                    
                    # Step 1: Get inspection findings first
                    findings_question = f"What are all the repair issues, defects, and problems found in this inspection report? Include HVAC, electrical, plumbing, structural, and safety issues."
                    
                    docs = self.retrievers['primary'].invoke(findings_question)
                    context = format_docs(docs)
                    
                    # Step 2: Extract key repair types from findings for cost estimates
                    repair_types_to_check = []
                    context_lower = context.lower()
                    
                    # Check for common repair categories found in inspection
                    if any(word in context_lower for word in ['hvac', 'heating', 'cooling', 'furnace', 'ac']):
                        repair_types_to_check.append("HVAC system")
                    if any(word in context_lower for word in ['electrical', 'wiring', 'panel', 'outlet']):
                        repair_types_to_check.append("electrical panel")
                    if any(word in context_lower for word in ['plumbing', 'pipe', 'leak', 'water']):
                        repair_types_to_check.append("plumbing repair")
                    if any(word in context_lower for word in ['roof', 'roofing', 'shingle']):
                        repair_types_to_check.append("roof repair")
                    if any(word in context_lower for word in ['floor', 'flooring']):
                        repair_types_to_check.append("flooring")
                    
                    # Step 3: Get cost estimates for identified repair types
                    cost_info = ""
                    if repair_types_to_check:
                        logger.info(f"💰 Getting cost estimates for: {repair_types_to_check}")
                        cost_info = "\n\n**Cost Estimates for Identified Issues:**\n"
                        
                        for repair_type in repair_types_to_check[:3]:  # Limit to 3 to avoid too many API calls
                            try:
                                # Use the same mapping logic as the notebook
                                # Simple mapping for common repair types
                                repair_type_lower = repair_type.lower()
                                if "hvac" in repair_type_lower:
                                    mapped_repair_type = "hvac_repair"
                                elif "electrical" in repair_type_lower:
                                    mapped_repair_type = "electrical_panel_replacement"
                                elif "plumbing" in repair_type_lower:
                                    mapped_repair_type = "plumbing_repair"
                                elif "roof" in repair_type_lower:
                                    mapped_repair_type = "roof_repair"
                                elif "floor" in repair_type_lower:
                                    mapped_repair_type = "flooring_installation"
                else:
                                    mapped_repair_type = "general_repair"
                                
                                api_url = f"https://repair-cost-api-618596951812.us-central1.run.app/api/v1/repair-cost/{mapped_repair_type}"
                                headers = {"x-api-key": os.getenv("REPAIR_API_KEY", "demo-key-2024")}
                                params = {"zip_code": "90210", "scope": "average"}
                                
                                import requests
                                response = requests.get(api_url, headers=headers, params=params, timeout=10)
                                
                                if response.status_code == 200:
                                    data = response.json()
                                    cost_est = data.get('cost_estimate', {})
                                    avg_cost = cost_est.get('average', 0)
                                    cost_info += f"- **{repair_type}**: ${avg_cost:,} average\n"
                else:
                                    cost_info += f"- **{repair_type}**: Cost estimate unavailable\n"
                            except Exception as e:
                                logger.error(f"Error getting cost for {repair_type}: {e}")
                                cost_info += f"- **{repair_type}**: Cost estimate unavailable\n"
                    
                    # Step 4: Provide comprehensive budget analysis
                    comprehensive_prompt = f"""Based on the inspection findings and cost estimates, {question}
                    
                    INSPECTION FINDINGS:
                    {context}
                    
                    {cost_info}
                    
                    Please provide prioritized repair recommendations considering:
                    1. Safety-critical issues (immediate priority)
                    2. Structural problems (high priority) 
                    3. Major systems (HVAC, electrical, plumbing)
                    4. Cosmetic improvements (lower priority)
                    5. The available budget mentioned in the question
                    
                    Format your response with specific budget allocation suggestions."""
                    
                    rag_prompt = ChatPromptTemplate.from_template("""
You are an expert property inspection analyst helping first-time home buyers with budget planning.

Provide specific, actionable recommendations based on the inspection findings and cost estimates.

QUESTION: {question}

Your response should include:
1. **Immediate Priorities** (safety issues)
2. **Budget Breakdown** (how to allocate the available funds)
3. **Timeline** (what to do first, second, etc.)
4. **Cost-Effective Solutions** (best value repairs)

Be specific about which repairs to prioritize within the budget constraints.
""")
                    
                    rag_chain = rag_prompt | self.llm | StrOutputParser()
                    rag_response = rag_chain.invoke({"question": comprehensive_prompt})
                    return {"messages": [AIMessage(content=f"**Budget Prioritization Analysis:**\n\n{rag_response}")]}
                else:
                    # Standard inspection findings question
                    docs = self.retrievers['primary'].invoke(question)
                    context = format_docs(docs)
                    
                    rag_prompt = ChatPromptTemplate.from_template("""
You are an expert property inspection analyst helping first-time home buyers understand their inspection report.

Your role is to analyze ONLY what the inspector actually found and documented in this specific inspection report.

INSTRUCTIONS:
- Base all responses strictly on the inspector's actual findings in the provided context
- Focus on specific issues, defects, or concerns the inspector identified
- Ignore boilerplate disclaimers, scope limitations, and general educational content
- If the inspector didn't find or comment on something, state that clearly
- For broad questions, synthesize the overall pattern of findings
- DO NOT speculate beyond what's documented

CONTEXT FROM INSPECTION REPORT:
{context}

QUESTION: {question}

Remember: You can only discuss what this inspector actually observed and documented.
""")
                    
                    rag_chain = rag_prompt | self.llm | StrOutputParser()
                    rag_response = rag_chain.invoke({"context": context, "question": question})
                    return {"messages": [AIMessage(content=f"**Inspection Analysis:**\n\n{rag_response}")]}

            # Build simplified graph with force_tools node
            workflow = StateGraph(AgentState)
            workflow.add_node("agent", agent_node)
            workflow.add_node("tools", tool_node)
            workflow.add_node("force_tools", force_tools_node)
            workflow.add_node("rag_analysis", rag_analysis_node)
            
            workflow.add_edge(START, "agent")
            workflow.add_conditional_edges(
                "agent", 
                should_continue,
                {
                    "tools": "tools",
                    "force_tools": "force_tools", 
                    "rag_analysis": "rag_analysis",
                    END: END
                }
            )
            workflow.add_edge("tools", END)  # Tools always end the conversation
            workflow.add_edge("force_tools", "tools")  # Force tools then go to tools node
            workflow.add_edge("rag_analysis", END)

            self.agent = workflow.compile()
            logger.info("Enhanced agent built successfully with multi-query retriever")
            
        except Exception as e:
            logger.error(f"Failed to build agent: {str(e)}")
            raise
    
    async def analyze_report(self, question: str) -> Dict[str, str]:
        """Analyze an inspection report with the enhanced agent"""
        try:
            initial_state = {
                "messages": [HumanMessage(content=question)],
                "question": question
            }
            
            result = self.agent.invoke(initial_state)
            
            # Extract the final answer from the last message
            final_message = result.get("messages", [])[-1] if result.get("messages") else None
            answer = final_message.content if final_message else "No answer generated"
            
            # Determine strategy based on the question content
            strategy = "Multi-Query RAG"
            if any(keyword in question.lower() for keyword in ['cost', 'price', 'estimate', 'budget']):
                strategy = "Tools + RAG"
            elif any(keyword in question.lower() for keyword in ['prioritize', 'urgent', 'critical']):
                strategy = "Complex RAG Analysis"
            
            return {
                "answer": answer,
                "strategy": strategy,
                "reasoning": "Enhanced agent with multi-query retriever and intelligent tool routing"
            }
            
        except Exception as e:
            logger.error(f"Report analysis error: {str(e)}")
            return {
                "answer": f"Analysis error: {str(e)}",
                "strategy": "ERROR",
                "reasoning": "Analysis failed due to technical error"
            }
    
    async def add_document(self, pdf_text: str, file_id: str):
        """Add a new document to the vector store"""
        try:
            from langchain.schema import Document
            
            # Create document
            document = Document(
                page_content=pdf_text,
                metadata={"source": file_id, "type": "inspection_report"}
            )
            
            # Split the document
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents([document])
            
            # Add to vector store
            self.vectorstore.add_documents(documents=splits)
            
            # Update retrievers with new content
            self.multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 8}),
                llm=self.llm
            )
            
            self.retrievers = {
                'baseline': self.vectorstore.as_retriever(search_kwargs={"k": 8}),
                'multi_query': self.multi_query_retriever,
                'primary': self.multi_query_retriever
            }
            
            logger.info(f"Successfully added document {file_id} to knowledge base")
            
        except Exception as e:
            logger.error(f"Failed to add document: {str(e)}")
            raise
    
    async def chat(self, question: str) -> Dict[str, str]:
        """Chat with the agent about inspection reports"""
        return await self.analyze_report(question)