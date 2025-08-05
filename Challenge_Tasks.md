## Property Inspection Report Analyzer for Southern California First-Time Home Buyers

### Task 1: Articulate the problem and the user of your application

First-time home buyers in Southern California receive overwhelming 50-200 page inspection reports that they cannot interpret, leading to poor purchase decisions and unexpected repair costs.


#### Why This is a Problem
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

The Property Inspection Report Analyzer will transform complex inspection documents into clear, actionable insights with specific cost estimates and prioritized repair timelines. Users will upload their inspection PDF and receive an easy to understand breakdown of issues categorized by urgency (immediate safety concerns, items to address within 6 months, and long-term maintenance), complete with Southern California-specific repair cost estimates and other recommendations.

1. Write 1-2 paragraphs on your proposed solution.  How will it look and feel to the user?
2. Describe the tools you plan to use in each part of your stack.  Write one sentence on why you made each tooling choice.
    1. LLM
    2. Embedding Model
    3. Orchestration
    4. Vector Database
    5. Monitoring
    6. Evaluation
    7. User Interface
    8. (Optional) Serving & Inference
3. Where will you use an agent or agents?  What will you use “agentic reasoning” for in your app?



Task 3: Collect data for (at least) RAG and choose (at least) one external API





RAGAS Evaluation Results:
============================================================
1st Multi-Query:
    Overall: 0.688
    Context Recall: 0.619
    Faithfulness: 0.843
    Answer Relevancy: 0.601

2nd Baseline:
    Overall: 0.636
    Context Recall: 0.500
    Faithfulness: 0.800
    Answer Relevancy: 0.610

3rd Parent-Document:
    Overall: 0.559
    Context Recall: 0.564
    Faithfulness: 0.652
    Answer Relevancy: 0.461

Best Performer: Multi-Query (0.688)
