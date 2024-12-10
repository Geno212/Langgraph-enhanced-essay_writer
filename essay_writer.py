from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLineEdit, QPushButton, QTextEdit, \
    QTabWidget, QLabel, QHBoxLayout, QSpinBox, QInputDialog, QMessageBox
from dotenv import load_dotenv

_ = load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
import operator
from langgraph.checkpoint.memory import MemorySaver  # Using MemorySaver instead of SqliteSaver
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from pydantic import BaseModel
from pydantic import ValidationError  # Import ValidationError

# Initialize MemorySaver and Tavily
memory = MemorySaver()
tavily = TavilyClient(api_key="")

# Define the Queries model to expect a list of queries as structured output
class Queries(BaseModel):
    queries: List[str]

# Define the state structure
class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int

    # System Prompts


PLAN_PROMPT = """You are an expert writer tasked with writing a high-level outline of an essay. 
 Write such an outline for the user-provided topic. Give an outline of the essay along with any 
 relevant notes or instructions for the sections."""

WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays.
 Generate the best essay possible for the user's request and the initial outline. Utilize all the 
 information below as needed:

 ------

 {content}"""

REFLECTION_PROMPT = """You are a teacher grading an essay submission. Generate critique and 
 recommendations for the user's submission. Provide detailed recommendations, including requests 
 for length, depth, style, etc."""

RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can be used 
 when writing the following essay. Generate a list of search queries that will gather any relevant 
 information. Only generate 3 queries max."""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can be 
 used when making any requested revisions (as outlined below). Generate a list of search queries 
 that will gather any relevant information. Only generate 3 queries max."""


class WorkflowThread(QThread):
    # Define a signal to update the UI from the worker thread
    update_tabs_signal = pyqtSignal(dict)
    save_state_signal = pyqtSignal(dict)

    def __init__(self, graph, initial_state, current_thread_id):
        super().__init__()
        self.graph = graph
        self.initial_state = initial_state
        self.current_thread_id = current_thread_id

    def run(self):
        """Run the workflow in a separate thread."""
        for state in self.graph.stream(self.initial_state, {"configurable": {"thread_id": self.current_thread_id}}):
            # Emit signals to update the GUI
            self.save_state_signal.emit(state)
            self.update_tabs_signal.emit(state)


class AdvancedEssayWriterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Essay Writer - Live State Viewer")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize the model
        self.model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0,
                                openai_api_key="")

        # Initialize Tavily and Thread States
        self.thread_states: Dict[str, List[dict]] = {}
        self.current_thread_id = None
        self.current_state_index = -1

        # Main Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Tabs for Workflow
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # Input and Settings Tab
        self.input_tab = QWidget()
        self.input_layout = QVBoxLayout(self.input_tab)
        self.task_input = QLineEdit()
        self.task_input.setPlaceholderText("Enter your essay topic...")
        self.input_layout.addWidget(self.task_input)
        self.generate_button = QPushButton("Start Workflow")
        self.input_layout.addWidget(self.generate_button)
        self.tabs.addTab(self.input_tab, "Input")

        # Plan, Draft, Critique, and Content Tabs
        self.plan_tab = self.create_tab("Plan")
        self.draft_tab = self.create_tab("Draft")
        self.critique_tab = self.create_tab("Critique")
        self.content_tab = self.create_tab("Content (Research)")

        # Controls Layout
        self.controls_layout = QHBoxLayout()
        self.thread_id_button = QPushButton("Set Thread ID")
        self.controls_layout.addWidget(self.thread_id_button)
        self.navigate_button = QPushButton("Navigate to Thread")
        self.controls_layout.addWidget(self.navigate_button)
        self.revision_spinbox = QSpinBox()
        self.revision_spinbox.setRange(1, 10)
        self.revision_spinbox.setValue(2)
        self.controls_layout.addWidget(QLabel("Max Revisions:"))
        self.controls_layout.addWidget(self.revision_spinbox)
        self.layout.addLayout(self.controls_layout)

        # Button Connections
        self.generate_button.clicked.connect(self.start_workflow)
        self.thread_id_button.clicked.connect(self.set_thread_id)
        self.navigate_button.clicked.connect(self.navigate_to_thread)

        # Initialize StateGraph
        self.graph = self.initialize_graph(AgentState)

    def create_tab(self, title):
        """Create a tab for displaying state information."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        display = QTextEdit()
        display.setReadOnly(True)
        layout.addWidget(QLabel(f"{title}:"))
        layout.addWidget(display)
        self.tabs.addTab(tab, title)
        return display


    def set_thread_id(self):
        """Prompt the user to set a thread ID."""
        thread_id, ok = QInputDialog.getText(self, "Set Thread ID", "Enter a unique thread ID:")
        if ok and thread_id:
            self.current_thread_id = thread_id
            if thread_id not in self.thread_states:
                self.thread_states[thread_id] = []
            QMessageBox.information(self, "Thread ID Set", f"Thread ID set to: {thread_id}")
        else:
            QMessageBox.warning(self, "Invalid Input", "Thread ID cannot be empty.")

    def save_state(self, state: dict):
        """Save the current state to the thread's state history."""
        if not self.current_thread_id:
            QMessageBox.warning(self, "Thread ID Required", "Please set a thread ID first.")
            return
        thread_history = self.thread_states[self.current_thread_id]
        thread_history.append(state.copy())
        self.current_state_index = len(thread_history) - 1
        print("Saved state:", state)  # Debug output

    def load_state(self, thread_id: str, index: int):
        """Load a specific state by thread ID and index."""
        if thread_id not in self.thread_states or index >= len(self.thread_states[thread_id]):
            QMessageBox.warning(self, "Invalid State", "The requested state does not exist.")
            return
        self.current_thread_id = thread_id
        state = self.thread_states[thread_id][index]
        self.update_tabs(state)
        self.test_update_tabs()

    def update_tabs(self, state: dict):
        """Update the tabs with the current state."""
        print("Updating tabs with state:", state)

        # Debugging output to check the state structure
        print("plan:", state.get("planner", {}).get("plan", "No plan found"))  # Safe access with .get()
        print("Draft:", state.get("generate", {}).get("draft", "No draft found"))
        print("Critique:", state.get("reflect", {}).get("critique", "No critique found"))
        print("Content:", state.get("research_plan", {}).get("content", "No content found"))
        print("Content after critique:", state.get("research_critique", {}).get("content", "No content found"))
        # Safely updating the tabs
        plan_text = state.get("planner", {}).get("plan", "")
        if plan_text:
            self.plan_tab.setPlainText(plan_text)

        content = state.get("research_plan", {}).get("content", [])
        if content:
            self.content_tab.setPlainText("\n".join(content))

        draft_text = state.get("generate", {}).get("draft", "")
        if draft_text:
            self.draft_tab.setPlainText(draft_text)

        critique_text = state.get("reflect", {}).get("critique", "")
        if critique_text:
            self.critique_tab.setPlainText(critique_text)

        content2=state.get("research_critique", {}).get("content", [])
        if content2:
            self.content_tab.setPlainText("\n".join(content2))

    # def test_update_tabs(self):
    #     mock_state = {
    #         "plan": "This is a test plan.",
    #         "draft": "This is a test draft.",
    #         "critique": "This is a test critique.",
    #         "content": ["Research on WWII", "Historical facts", "Key events"]
    #     }
    #     self.update_tabs(mock_state)
    #     print("heloooooooo")

    def navigate_to_thread(self):
        """Navigate to a specific thread ID and state index."""
        thread_id, ok = QInputDialog.getText(self, "Navigate by Thread ID", "Enter the thread ID:")
        if ok and thread_id:
            if thread_id in self.thread_states:
                index, ok = QInputDialog.getInt(self, "Select State", "Enter the state index:", 0, 0,
                                                len(self.thread_states[thread_id]) - 1)
                if ok:
                    self.load_state(thread_id, index)
            else:
                QMessageBox.warning(self, "Invalid Thread ID", f"No states found for Thread ID: {thread_id}")

    def start_workflow(self):
        """Start the essay writing workflow in a separate thread."""
        if not self.current_thread_id:
            QMessageBox.warning(self, "Thread ID Required", "Please set a thread ID first.")
            return

        task = self.task_input.text()
        if not task:
            QMessageBox.warning(self, "Input Required", "Please enter a topic.")
            return

        # Initial State
        istate = {
            "task": task,
            "plan": "",
            "draft": "",
            "critique": "",
            "content": [],
            "revision_number": 1,
            "max_revisions": self.revision_spinbox.value()
        }

        # Create and start the workflow thread
        self.workflow_thread = WorkflowThread(self.graph, istate, self.current_thread_id)

        # Connect the signals to update the UI and save state
        self.workflow_thread.update_tabs_signal.connect(self.update_tabs)
        self.workflow_thread.save_state_signal.connect(self.save_state)

        # Start the thread
        self.workflow_thread.start()

    def initialize_graph(self, AgentState):
        """Initialize the LangGraph workflow."""
        builder = StateGraph(AgentState)
        builder.add_node("planner", self.plan_node)
        builder.add_node("generate", self.generation_node)
        builder.add_node("reflect", self.reflection_node)
        builder.add_node("research_plan", self.research_plan_node)
        builder.add_node("research_critique", self.research_critique_node)

        # Set entry point and conditional edges
        builder.set_entry_point("planner")
        builder.add_conditional_edges(
            "generate",
            self.should_continue,
            {END: END, "reflect": "reflect"}
        )

        builder.add_edge("planner", "research_plan")
        builder.add_edge("research_plan", "generate")
        builder.add_edge("reflect", "research_critique")
        builder.add_edge("research_critique", "generate")

        return builder.compile(checkpointer=memory)

    # Node Functions (same as before)
    def plan_node(self, state: AgentState):
        messages = [
            SystemMessage(content=PLAN_PROMPT),
            HumanMessage(content=state['task'])
        ]
        response = self.model.invoke(messages)
        return {"plan": response.content}

    def generation_node(self, state: AgentState):
        content = "\n\n".join(state['content'] or [])
        user_message = HumanMessage(
            content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}"
        )
        messages = [
            SystemMessage(content=WRITER_PROMPT.format(content=content)),
            user_message
        ]
        response = self.model.invoke(messages)
        return {
            "draft": response.content,
            "revision_number": state.get("revision_number", 1) + 1
        }

    def reflection_node(self, state: AgentState):
        messages = [
            SystemMessage(content=REFLECTION_PROMPT),
            HumanMessage(content=state['draft'])
        ]
        response = self.model.invoke(messages)
        return {"critique": response.content}

    def research_plan_node(self, state: AgentState):
        queries = self.model.with_structured_output(Queries).invoke([
            SystemMessage(content=RESEARCH_PLAN_PROMPT),
            HumanMessage(content=state['task'])
        ])
        content = state['content'] or []
        print(queries)
        for q in queries.queries:
            response = tavily.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])
        return {"content": content}

    def research_critique_node(self, state: AgentState):
        queries = self.model.with_structured_output(Queries).invoke([
            SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=state['critique'])
        ])
        content = state['content'] or []
        print(queries)
        for q in queries.queries:
            response = tavily.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])
        return {"content": content}

    def should_continue(self, state):
        if state["revision_number"] > state["max_revisions"]:
            return END
        return "reflect"


if __name__ == "__main__":
    app = QApplication([])
    gui = AdvancedEssayWriterGUI()
    gui.show()
    app.exec_()
