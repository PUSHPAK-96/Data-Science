
# Automated Survey Analysis Tool

An interactive NLP-powered tool that automatically processes survey responses, performs sentiment analysis, extracts insights, and presents results through a Streamlit dashboard.

## ✅ Objective
To build an intelligent system that:
- Processes large survey datasets  
- Performs sentiment analysis  
- Extracts key insights  
- Visualizes data interactively  
- Helps businesses make data-driven decisions  

## ✅ Tools & Technologies
- Python  
- Streamlit  
- Pandas  
- Scikit-Learn  
- VADER Sentiment Analysis  
- TF-IDF Keyword Extraction  
- Altair Charts  
- CSV Dataset  

## ✅ Project Structure
```
survey-insights/
│
├── app.py
├── requirements.txt
│
├── src/
│   ├── prep.py
│   ├── sentiment.py
│   ├── keywords.py
│   ├── topics.py
│   └── __init__.py
│
├── data/
│   ├── sample_surveys.csv
│
└── README.md
```

## ✅ Workflow
1. Load & clean dataset  
2. Sentiment scoring  
3. Sentiment labeling  
4. Keyword extraction  
5. Optional topic modeling  
6. Dashboard visualization  
7. Export enriched CSV  

## ✅ How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## ✅ Deployment
### Streamlit Cloud  
1. Push repo to GitHub  
2. Go to share.streamlit.io  
3. Select repo → app.py → Deploy  

### Hugging Face Spaces  
1. Create Space  
2. Select Streamlit SDK  
3. Upload repo  

## ✅ Conclusion
- Automates survey analytics  
- Interactive & scalable  
- Useful for businesses, HR, marketing, product teams  
