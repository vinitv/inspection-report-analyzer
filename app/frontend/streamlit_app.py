import streamlit as st
import requests
import time
from typing import Dict, Any
import json
import re

class InspectionReportApp:
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.setup_page_config()
        self.setup_custom_css()
        
    def setup_page_config(self):
        """Configure Streamlit page settings"""
st.set_page_config(
            page_title="Property Inspection Analyzer",
    page_icon="🏠",
    layout="wide",
            initial_sidebar_state="collapsed"
        )
    
    def setup_custom_css(self):
        """Apply modern custom CSS styling"""
st.markdown("""
<style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global styling */
    .main .block-container {
            padding: 1rem 2rem;
            max-width: 100%;
            font-family: 'Inter', sans-serif;
    }
    
        /* Header */
        .app-header {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
        padding: 2rem;
            border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
            color: white;
            box-shadow: 0 10px 40px rgba(99, 102, 241, 0.2);
        }
        
        .app-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .app-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.95;
            font-weight: 400;
        }
        
        /* Left panel styling */
        .left-panel {
            background: white;
            border-radius: 16px;
        padding: 1.5rem;
            box-shadow: 0 4px 24px rgba(0,0,0,0.06);
            border: 1px solid #f1f5f9;
            height: fit-content;
            position: sticky;
            top: 1rem;
        }
        
        /* Upload section */
        .upload-section {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border: 2px dashed #cbd5e1;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
        }
        
        .upload-section:hover {
            border-color: #6366f1;
            background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
        }
        
        .upload-success {
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
            border: 2px solid #34d399;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            text-align: center;
        }
        
        /* Sample questions */
        .questions-section h3 {
            color: #1e293b;
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .question-category {
            margin-bottom: 1rem;
        }
        
        .category-title {
            font-size: 0.875rem;
            font-weight: 600;
            color: #6366f1;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .sample-question {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            cursor: pointer;
            transition: all 0.2s ease;
            font-size: 0.9rem;
            color: #475569;
            text-align: left;
        }
        
        .sample-question:hover {
            background: #6366f1;
            color: white;
            border-color: #6366f1;
            transform: translateX(4px);
        }
        
        /* Right panel - Chat */
        .chat-panel {
            background: white;
            border-radius: 16px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.06);
            border: 1px solid #f1f5f9;
            height: 700px;
            display: flex;
            flex-direction: column;
        }
        
        .chat-header {
            padding: 1.5rem;
            border-bottom: 1px solid #f1f5f9;
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border-radius: 16px 16px 0 0;
        }
        
        .chat-header h3 {
            margin: 0;
            color: #1e293b;
            font-size: 1.2rem;
            font-weight: 600;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
        padding: 1rem;
            background: #fafbfc;
        }
        
        .chat-input-area {
            padding: 1rem;
            border-top: 1px solid #f1f5f9;
            background: white;
            border-radius: 0 0 16px 16px;
        }
        
        /* Message styling */
        .message {
            margin-bottom: 1rem;
            max-width: 85%;
            word-wrap: break-word;
        }
        
        .user-message {
            margin-left: auto;
            margin-right: 0;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
            padding: 1rem 1.25rem;
            border-radius: 18px 18px 6px 18px;
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
        }
        
        .ai-message {
            margin-left: 0;
            margin-right: auto;
            background: white;
            color: #374151;
            padding: 1rem 1.25rem;
            border-radius: 18px 18px 18px 6px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        
        .message-content {
            line-height: 1.6;
            font-size: 0.95rem;
        }
        
        /* Markdown styling in messages */
        .ai-message h3 {
            color: #1f2937;
            font-size: 1.1rem;
            font-weight: 600;
            margin: 0 0 0.75rem 0;
            border-bottom: 2px solid #6366f1;
            padding-bottom: 0.25rem;
        }
        
        .ai-message h4 {
            color: #374151;
            font-size: 1rem;
            font-weight: 600;
            margin: 1rem 0 0.5rem 0;
        }
        
        .ai-message ul {
            margin: 0.5rem 0;
            padding-left: 1.5rem;
        }
        
        .ai-message li {
            margin-bottom: 0.25rem;
            line-height: 1.5;
        }
        
        .ai-message strong {
            color: #1f2937;
            font-weight: 600;
        }
        
        /* Cost highlighting */
        .cost-highlight {
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            color: #92400e;
            padding: 0.2rem 0.4rem;
            border-radius: 6px;
            font-weight: 600;
            border: 1px solid #f59e0b;
        }
        
        /* Priority badges */
        .priority-high {
            background: #dc2626;
            color: white;
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .priority-medium {
            background: #f59e0b;
            color: white;
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .priority-low {
            background: #059669;
            color: white;
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Empty state */
        .empty-chat {
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            padding: 2rem;
            color: #6b7280;
        }
        
        .empty-chat h3 {
            color: #374151;
            margin-bottom: 0.5rem;
        }
        
        /* Hide Streamlit elements */
        .stDeployButton { display: none; }
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        .stAppHeader { display: none; }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            font-size: 0.9rem;
            transition: all 0.2s ease;
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
        }
        
        /* File uploader styling */
        .stFileUploader {
            background: transparent;
        }
        
        /* Chat input styling */
        .stChatInput > div > div > textarea {
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            padding: 0.75rem;
            font-size: 0.95rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render the main application header"""
        st.markdown("""
        <div class="app-header">
            <h1>🏠 Property Inspection Analyzer</h1>
            <p>AI-powered analysis with real-time cost estimation and smart prioritization</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_left_panel(self):
        """Render the left panel with upload and sample questions"""
        with st.container():
            st.markdown('<div class="left-panel">', unsafe_allow_html=True)
            
            # Upload section
            if not st.session_state.get("report_processed", False):
                st.markdown("""
                <div class="upload-section">
                    <h3 style="margin: 0 0 1rem 0; color: #475569;">📄 Upload Report</h3>
                    <p style="margin: 0; color: #64748b; font-size: 0.9rem;">Upload your property inspection report to get started</p>
                </div>
                """, unsafe_allow_html=True)
                
                uploaded_file = st.file_uploader(
                    "Choose a PDF file",
                    type=['pdf'],
                    help="Upload your property inspection report (PDF format only)",
                    label_visibility="collapsed"
                )
                
                if uploaded_file is not None:
                    if st.button("🔍 Analyze Report", use_container_width=True):
                        with st.spinner("Processing your inspection report..."):
                            success = self.upload_and_process_file(uploaded_file)
                            if success:
                                st.session_state.report_processed = True
                                st.rerun()
            else:
                st.markdown("""
                <div class="upload-success">
                    <h3 style="margin: 0 0 0.5rem 0; color: #065f46;">✅ Report Ready</h3>
                    <p style="margin: 0; color: #047857; font-size: 0.9rem;">Your inspection report has been processed</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Sample questions
            st.markdown("""
            <div class="questions-section">
                <h3>💡 Sample Questions</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Question categories
            questions = {
                "💰 Cost Estimates": [
                    "What's the cost of HVAC repair?",
                    "How much does electrical work cost?",
                    "What are plumbing repair costs?"
                ],
                "📊 Budget Planning": [
                    "What should I prioritize with $5,000?",
                    "What repairs are most urgent?",
                    "What can I do for under $1,000?"
                ],
                "🏠 Inspection Issues": [
                    "What electrical issues were found?",
                    "Are there any HVAC problems?",
                    "What structural concerns exist?"
                ],
                "⚠️ Safety & Priority": [
                    "What safety hazards exist?",
                    "Which repairs are critical?",
                    "What needs immediate attention?"
                ]
            }
            
            for category, question_list in questions.items():
                st.markdown(f'<div class="category-title">{category}</div>', unsafe_allow_html=True)
                for question in question_list:
                    if st.button(question, key=f"q_{question}", use_container_width=True):
                        self.handle_question_click(question)
            
            # Clear session button
            if st.session_state.get("report_processed", False):
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🗑️ New Report", use_container_width=True):
                    self.clear_session()
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

    def render_chat_panel(self):
        """Render the right panel with chat interface"""
        st.markdown('<div class="chat-panel">', unsafe_allow_html=True)
        
        # Chat header
        st.markdown("""
        <div class="chat-header">
            <h3>💬 Chat with AI Assistant</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.get("report_processed", False):
            # Empty state
            st.markdown("""
            <div class="empty-chat">
                <h3>Ready to analyze your report</h3>
                <p>Upload an inspection report to start asking questions about the property.</p>
                <br>
                <p><strong>You can ask about:</strong></p>
                <ul style="text-align: left; max-width: 300px;">
                    <li>Repair costs and estimates</li>
                    <li>Budget prioritization</li>
                    <li>Safety concerns</li>
                    <li>System conditions</li>
                    <li>Structural issues</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Chat messages container
            st.markdown('<div class="chat-messages" id="chat-messages">', unsafe_allow_html=True)
            
            # Display messages
            if "messages" in st.session_state:
                for message in st.session_state.messages:
                    self.render_message(message)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Chat input
            st.markdown('<div class="chat-input-area">', unsafe_allow_html=True)
            if prompt := st.chat_input("Ask about your inspection report..."):
                self.handle_user_message(prompt)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    def render_message(self, message):
        """Render a single chat message"""
        if message["role"] == "user":
                    st.markdown(f"""
            <div class="message user-message">
                <div class="message-content">{message["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
            formatted_content = self.format_ai_response(message["content"])
            st.markdown(f"""
            <div class="message ai-message">
                <div class="message-content">{formatted_content}</div>
            </div>
            """, unsafe_allow_html=True)

    def format_ai_response(self, content: str) -> str:
        """Format AI response with proper HTML and styling"""
        if not content:
            return "No response available."
        
        # Convert markdown-style formatting to HTML
        
        # Headers - handle both ** and ### patterns
        content = re.sub(r'### ([^\n]+)', r'<h3>\1</h3>', content)
        content = re.sub(r'\*\*([^*\n]+):\*\*', r'<h3>\1</h3>', content)
        content = re.sub(r'\*\*([^*\n]+)\*\*', r'<strong>\1</strong>', content)
        
        # Convert bullet points and numbered lists
        lines = content.split('\n')
        formatted_lines = []
        in_list = False
        list_type = None
            
            for line in lines:
            line = line.strip()
            if line.startswith('- ') or line.startswith('• '):
                if not in_list or list_type != 'ul':
                    if in_list:
                        formatted_lines.append(f'</{list_type}>')
                    formatted_lines.append('<ul>')
                    in_list = True
                    list_type = 'ul'
                formatted_lines.append(f'<li>{line[2:]}</li>')
            elif re.match(r'^\d+\.\s', line):
                if not in_list or list_type != 'ol':
                    if in_list:
                        formatted_lines.append(f'</{list_type}>')
                    formatted_lines.append('<ol>')
                    in_list = True
                    list_type = 'ol'
                formatted_lines.append(f'<li>{re.sub(r"^\d+\.\s", "", line)}</li>')
            else:
                if in_list:
                    formatted_lines.append(f'</{list_type}>')
                    in_list = False
                    list_type = None
                if line:
                    formatted_lines.append(f'<p>{line}</p>')
                elif not in_list:
                    formatted_lines.append('<br>')
        
        if in_list:
            formatted_lines.append(f'</{list_type}>')
        
        content = ''.join(formatted_lines)
        
        # Highlight costs
        cost_pattern = r'(\$[\d,]+(?:\.\d{2})?)'
        content = re.sub(cost_pattern, r'<span class="cost-highlight">\1</span>', content)
        
        # Priority badges
        content = re.sub(r'\b(High Priority|Critical|Urgent)\b', r'<span class="priority-high">\1</span>', content, flags=re.IGNORECASE)
        content = re.sub(r'\b(Medium Priority|Moderate)\b', r'<span class="priority-medium">\1</span>', content, flags=re.IGNORECASE)
        content = re.sub(r'\b(Low Priority|Minor)\b', r'<span class="priority-low">\1</span>', content, flags=re.IGNORECASE)
        
        return content

    def handle_question_click(self, question: str):
        """Handle when a sample question is clicked"""
        if not st.session_state.get("report_processed", False):
            st.warning("Please upload an inspection report first.")
            return
        
        # Initialize messages if needed
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Get AI response
        response = self.get_ai_response(question)
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()

    def handle_user_message(self, message: str):
        """Handle user chat input"""
        # Initialize messages if needed
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": message})
        
        # Get AI response
        with st.spinner("Thinking..."):
            response = self.get_ai_response(message)
            if response:
                st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()

    def upload_and_process_file(self, uploaded_file) -> bool:
        """Upload and process the inspection report"""
        try:
            files = {"file": (uploaded_file.name, uploaded_file.read(), "application/pdf")}
            response = requests.post(f"{self.backend_url}/upload", files=files, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.session_id = result.get("session_id")
                return True
            else:
                st.error(f"Upload failed: {response.text}")
                return False
                
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")
            return False

    def get_ai_response(self, question: str) -> str:
        """Get response from AI backend"""
        try:
            if "session_id" not in st.session_state:
                return "Please upload an inspection report first."
            
            payload = {
                "question": question,
                "session_id": st.session_state.session_id
            }
            
            response = requests.post(
                f"{self.backend_url}/chat",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("answer", "No response generated.")
            else:
                return f"Error: {response.text}"
                
        except Exception as e:
            return f"Error: {str(e)}"

    def clear_session(self):
        """Clear current session data"""
        keys_to_clear = ['session_id', 'messages', 'report_processed']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
    
    def run(self):
        """Main application entry point"""
        self.render_header()
        
        # Create main layout
        col1, col2 = st.columns([1, 2], gap="large")
        
        with col1:
            self.render_left_panel()
        
        with col2:
            self.render_chat_panel()

if __name__ == "__main__":
    app = InspectionReportApp()
    app.run()