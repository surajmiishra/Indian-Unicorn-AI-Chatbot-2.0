# Indian Unicorn Chatbot 

A Python-based AI chatbot that answers questions about **Indian Unicorn startups** using a **static dataset**.  
It features **context-aware conversations**, **ambiguity handling**, and **production-grade engineering practices** such as logging, sanitization, and observability.

---

##  Setup & Execution

### Prerequisites

- Python 3.8+
- Pandas

Install Pandas:
```bash
pip install pandas
````

---

### Installation

1. Clone this repository or download the project files.
2. Ensure the dataset file `tracxn.csv` is present in the root directory.
3. Install dependencies:

   ```bash
   pip install pandas
   ```

---

### Running the Bot

Execute the main script:

```bash
python chatbot.py
```

To exit the chatbot, type:

```text
exit
```

or

```text
quit
```

---

##  Architecture & Design

### 1. Hybrid Retrieval Engine (`DataEngine`)

Instead of using simple keyword matching, the chatbot employs a **tiered retrieval strategy**:

#### Tier 1: Named Entity Recognition (NER) – O(1) Lookup

* Detects explicit company mentions (e.g., *Razorpay*)
* Directly retrieves the specific company record
* Bypasses unnecessary dataset scanning

#### Tier 2: Sector Mapping

* Maps broad user intents such as *Fintech* to multiple dataset columns:

  * Payments
  * Banking Tech
  * Alternative Lending
* Improves recall without using embeddings

#### Tier 3: Contextual Filtering

* Applies filters on the **previous query result**
* Enables natural follow-up questions like:

  ```
  Fintech → Bangalore → High Valuation
  ```
* Prevents unrelated results from appearing

---

### 2. State Management (`Chatbot`)

The chatbot maintains:

* Conversation history
* A persistent `current_context_df` representing the last query result

This enables **multi-step drill-down queries** without re-processing the entire dataset each time.

---

### 3. Engineering Practices

#### Sanitization

* Regex-based input cleaning
* Prevents malformed queries or unsafe input

#### Resilience

* Automatic CSV encoding detection (UTF-8 / Latin-1)
* Ensures stability when handling imperfect datasets

#### Observability

* Singleton `MetricsTracker` captures:

  * Query latency
  * Error rates
  * Clarification triggers

#### Logging

* Structured logs written to `chatbot.log`
* Aids debugging and performance analysis

---

##  Sample Interaction

```
User: Tell me about fintech unicorns.
Bot: Found the following companies:
     - Juspay (Payments)
     - Money View (Alternative Lending)

User: Which of these are based in Bangalore?
Bot: Based on your previous query, the companies matching your criteria are:
     Juspay, Money View, Perfios, Open, ACKO
(Note: Correctly excludes non-fintech Bangalore companies like Rapido)

User: Which company is best to collaborate with?
Bot: Could you verify the criteria you are looking for?
     (e.g., Valuation, Sector, Location)
```

---

##  Assumptions & Limitations

* **Data Source**
  The chatbot relies entirely on the provided `tracxn.csv` dataset.

* **LLM Simulation**
  To allow immediate execution without API keys:

  * The `LLMClient` uses robust rule-based logic
  * Simulates intelligent responses
  * Can be easily replaced with `openai.ChatCompletion` inside the `generate_response` method

---


