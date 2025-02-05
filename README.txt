# Port Activity Monitor

A sophisticated data collection and analysis system designed to monitor vessel movements, delays, and port activities at the Port of Piraeus. This system serves as a data preparation component for advanced fleet management and route optimization applications.

## 🚢 Features

### Data Collection
- Real-time vessel tracking
- Expected arrivals monitoring
- Port calls (arrivals/departures) logging
- Automated data collection every 15 minutes
- Historical data maintenance (30-day retention)

### Data Processing
- Comprehensive data cleaning and validation
- Duplicate detection and removal
- Missing value handling
- Standardized data formats
- Quality assurance checks

### Analytics
- Port activity patterns analysis
- Vessel type statistics
- Delay tracking and analysis
- Data continuity monitoring

## 🛠 Technical Stack
- Python 3.x
- FastAPI
- MongoDB
- Playwright (web scraping)
- Pandas (data processing)
- Schedule (task automation)

## 📦 Installation

1. Clone the repository
```bash
git clone [https://github.com/CruxTheStaff/port_delays_monitor]

2. ##Create and activate virtual environment
BASH
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

3. ##Install dependencies
BASH

pip install -r requirements.txt

4. ##Install Playwright browsers
BASH

playwright install


###🚀 Usage
1. Start the data collection scheduler:
BASH

python scheduler.py


2. Run data cleaning:
BASH

python src/data_collection/data_cleaner.py


###📁 Project Structure

.
├── src/
│   ├── analytics/         # Analysis modules
│   ├── api/              # API endpoints
│   ├── config/           # Configuration files
│   ├── database/         # Database operations
│   ├── data_collection/  # Data collection and cleaning
│   ├── models/           # Data models
│   ├── scrapers/         # Web scraping modules
│   └── verification/     # Data verification
├── tests/                # Test files
├── main.py              # Main application
├── scheduler.py         # Data collection scheduler
└── requirements.txt     # Project dependencies


###📊 Data Types
##Current Vessels
Vessel identification
Technical specifications (DWT, GRT, size)
Arrival timestamps
Vessel type classification
##Expected Arrivals
MMSI tracking
Vessel details
Expected arrival times
##Port Calls
Arrival/Departure events
Timestamps
Vessel information


###🔄 Development Roadmap
##Current Phase
Port of Piraeus data collection and analysis
Data cleaning and validation
Basic analytics implementation
##Upcoming Features
Weather data integration
Advanced analytics
Peak time analysis
Vessel delay correlation
Port congestion predictions
Additional ports integration

## 📝 License
MIT License

Copyright (c) 2024 Stavroula Kamini

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

## 🤝 Contributing
This is a personal portfolio project developed for educational and demonstration purposes. While this project is primarily for personal use, feedback and suggestions are welcome through issues or pull requests.

## 📫 Contact
- LinkedIn: https://www.linkedin.com/in/stavroula-kamini/
- Email: cruxthestaff@helicondata.com
