🛫 **FlightFinder LLM**  
Find the best flights effortlessly using large language models and real-time flight data

📌 **Overview**  
FlightFinder LLM uses advanced large language models (LLMs) combined with live flight data to understand your travel needs and find the best flights tailored just for you. Whether it’s a quick business trip or a long vacation, simply tell FlightFinder LLM your departure, destination, dates, and preferences, and it will handle the rest — searching, filtering, sorting, and delivering personalized flight options in an easy-to-understand way.

🔥 **Example Workflow:**  
1. LLM agents understand user inputs — destination, departure city, travel dates, number of days, and preferences.  
2. Calls Google Flights API (or similar) to fetch live flight details.  
3. Applies smart sorting and filtering functions to find the best flights (price, duration, layovers, etc.).  
4. Uses LLM-powered wrapper to reword and summarize the flight options for easy user comprehension.

🚀 **Features**  
- **Natural Language Understanding:** Input travel plans conversationally, no rigid forms required.  
- **Multi-Agent Coordination:** Separate LLM agents parse user needs and reformat results.  
- **Real-Time Flight Search:** Integrates with flight data APIs to fetch up-to-date flight options.  
- **Custom Sorting Algorithms:** Finds flights based on price, duration, or convenience.  
- **Personalized Summaries:** Results are wrapped and explained by an LLM for clarity.  

🛠 **How it works**  
1. User inputs travel details in natural language.  
2. LLM agents extract key info: departure, destination, dates, stay duration.  
3. FlightFinder LLM queries flight APIs to get available flights.  
4. Internal functions rank flights by criteria like cost, duration, and number of stops.  
5. An LLM agent rewrites results into a user-friendly summary with recommendations.

▶️ **Usage**  
1. Run the main script:  
```bash  
python main.py  
```
2- Follow the prompt to enter your travel details naturally (e.g., "Find me flights from NYC to Paris for 5 days next month").
3- Receive a sorted, easy-to-read list of best flight options.

## 📚 Technologies Used
- **OpenAI GPT (LLM)** – For understanding user intent and summarizing results
- **Flight APIs** – Google Flights for live flight data retrieval
- **Requests** – For API calls
- **Custom Sorting Algorithms** – To find and rank optimal flights
