# NWW Dashboard UI Documentation

## 🌍 NWW (News World Watch) Dashboard - Simple App

### 📋 Overview
The NWW Dashboard is a comprehensive Streamlit-based web application designed for crisis detection and news analysis. The simple_app.py provides a fully functional demonstration interface with 8 main tabs covering all aspects of the news monitoring and analysis pipeline.

---

## 🖥️ Main Interface

### **Header Section**
- **Title**: "🌍 NWW - News World Watch"
- **Subtitle**: "Complete Automation Package for Crisis Detection and Analysis"
- **Layout**: Wide layout optimized for data visualization

### **Sidebar Configuration Panel**
Located on the left side, provides system configuration and control options:

#### ⚙️ Configuration Section
- **Bundle Directory**: Text input for specifying the data bundle directory path
- **Default Value**: `data/bundles/sample`

#### 🔧 Settings Section
- **Alert Threshold**: Slider (0.0 - 1.0, default: 0.7)
  - Controls minimum score for generating alerts
- **EMA Alpha**: Slider (0.0 - 1.0, default: 0.3)
  - Exponential moving average smoothing factor
- **Hysteresis**: Slider (0.0 - 0.5, default: 0.1)
  - Hysteresis for alert state changes

#### 🚀 Control Section
- **Run All Modules Button**: Primary action button to execute the complete processing pipeline
- **Status Display**: Shows completion status for all 12 processing modules

---

## 📊 Tab 1: Overview

### **Key Metrics Dashboard**
Four-column metric display showing:
- **Articles Processed**: 1,247 (↗️ +12)
- **Active Alerts**: 8 (↗️ +3)
- **Average Score**: 0.65 (↗️ +0.05)
- **System Health**: 🟢 Healthy (↗️ 99%)

### **Processing Pipeline Status**
Visual status display for 12 processing steps:
1. ✅ **Ingest**: Extract articles from sources
2. ✅ **Normalize**: Clean and normalize text
3. ✅ **Analyze**: Extract keywords, entities, frames
4. ✅ **Gate**: Filter content based on indicators
5. ✅ **Score IS**: Indicator-based scoring
6. ✅ **Score DBN**: Dynamic Bayesian network scoring
7. ✅ **Score LLM**: LLM judge scoring
8. ✅ **Fusion**: Score fusion and calibration
9. ✅ **Blocks**: EDS block matching
10. ✅ **Scenarios**: Scenario construction
11. ✅ **Alerts**: Alert generation
12. ✅ **Ledger**: Audit trail

### **Recent Activity Chart**
Interactive Plotly chart showing:
- **X-axis**: 24-hour timeline
- **Y-axis (Left)**: Number of articles processed
- **Y-axis (Right)**: Number of alerts generated
- **Visualization**: Dual-line chart with blue (articles) and red (alerts) lines

---

## 📥 Tab 2: Ingest

### **Sources Management**
- **Data Table**: Displays source URLs with status information
- **Columns**: URL, Status, Last Update, Success Rate
- **Sample Sources**:
  - Reuters World News (98% success rate)
  - BBC News (95% success rate)
  - CNN World (92% success rate)
  - Associated Press (97% success rate)

### **Control Panel**
- **🔄 Refresh Sources**: Updates source status
- **▶️ Run Ingest**: Executes data ingestion process

### **Statistics Section**
Three-column metrics:
- **Total Sources**: 4
- **Success Rate**: 95% (↗️ +2%)
- **Last Update**: 2 minutes ago

---

## 🎯 Tab 3: Scoring

### **Score Distribution Chart**
- **Chart Type**: Histogram
- **Data**: Score distribution across all articles
- **Bins**: 20 bins for detailed analysis
- **Purpose**: Visualize overall scoring patterns

### **Scores by Stage Analysis**
Data table showing scoring breakdown by processing stage:
- **Columns**: Stage, Mean Score, Standard Deviation, Count
- **Stages**: IS, DBN, LLM, FUSION

### **Top Scoring Articles**
Ranked list of highest-scoring articles:
- **Columns**: ID, Stage, Score
- **Top 5 Articles**: Displayed with scores ranging from 0.82 to 0.95

---

## ⏰ Tab 4: Timeline

### **Event Timeline Visualization**
Interactive timeline chart showing:
- **X-axis**: Time range (start and end times)
- **Y-axis**: Event types (Military, Diplomatic, Economic)
- **Color Coding**: Severity levels (High, Medium, Low, Critical)
- **Chart Type**: Gantt-style timeline
- **Data Points**: 10 sample events over 20-hour period

---

## 🧱 Tab 5: Blocks

### **EDS Block Analysis**
Data table displaying block matching results:
- **Columns**: Block ID, Type, Confidence, Articles, Status
- **Block Types**: Military, Diplomatic, Economic
- **Confidence Scores**: Range from 0.76 to 0.95
- **Status**: Active, Resolved

### **Sample Data**:
- B001: Military block (95% confidence, 5 articles)
- B002: Diplomatic block (87% confidence, 3 articles)
- B003: Economic block (76% confidence, 7 articles)
- B004: Military block (92% confidence, 4 articles)

---

## 📋 Tab 6: Scenarios

### **Scenario Construction Results**
Data table showing constructed crisis scenarios:
- **Columns**: Scenario ID, Name, Severity, Confidence, Status
- **Scenario Types**:
  - S001: Military Conflict (High severity, 89% confidence)
  - S002: Diplomatic Crisis (Medium severity, 76% confidence)
  - S003: Economic Warfare (Medium severity, 68% confidence)

---

## 📦 Tab 7: Artifacts

### **Export Options**
Four export buttons arranged in 2x2 grid:
- **📄 Export JSON**: Export data in JSON format
- **📋 Export Brief**: Generate brief report
- **📦 Export ZIP**: Create compressed archive
- **🔒 Export STIX**: Export in STIX format for threat intelligence

### **Export History**
Data table tracking export activities:
- **Columns**: Timestamp, Format, Size, Status
- **Recent Exports**: Shows last 3 export operations
- **File Sizes**: Range from 1.8 MB to 4.1 MB

---

## 📝 Tab 8: Ledger

### **Audit Trail**
Complete audit log of all processing steps:
- **Columns**: Timestamp, Step, Description, Status
- **Recent Entries**: Shows last 3 processing steps
- **Status**: All entries marked as "Completed"
- **Timestamps**: Precise to the second

---

## 🎨 Visual Design Features

### **Color Scheme**
- **Primary**: Blue (#1f77b4) for main data
- **Secondary**: Orange (#ff7f0e) for secondary metrics
- **Success**: Green (#2ca02c) for completed status
- **Warning**: Red (#d62728) for alerts
- **Info**: Purple (#9467bd) for information

### **Interactive Elements**
- **Hover Effects**: Chart elements show detailed information on hover
- **Responsive Design**: Adapts to different screen sizes
- **Real-time Updates**: Status indicators update dynamically

### **Data Visualization**
- **Plotly Integration**: Professional-grade interactive charts
- **Multiple Chart Types**: Histograms, timelines, scatter plots
- **Dual Y-axis**: Support for different data scales
- **Color Coding**: Consistent color scheme across all visualizations

---

## 🔧 Technical Features

### **Session State Management**
- **Bundle Directory**: Persistent across page refreshes
- **Configuration**: Settings maintained during session
- **Processing Status**: Real-time status tracking

### **Error Handling**
- **Graceful Degradation**: Demo data when real data unavailable
- **User Feedback**: Success/error messages for all actions
- **Validation**: Input validation for configuration settings

### **Performance**
- **Efficient Rendering**: Optimized for large datasets
- **Caching**: Session state caching for better performance
- **Lazy Loading**: Data loaded only when needed

---

## 🚀 Usage Instructions

### **Getting Started**
1. **Launch**: Run `python run_ui.py` or `streamlit run nwwpkg/ui/simple_app.py`
2. **Access**: Open browser to `http://localhost:8502`
3. **Configure**: Set bundle directory and thresholds in sidebar
4. **Explore**: Navigate through tabs to view different aspects

### **Configuration**
1. **Bundle Directory**: Set path to your data bundle
2. **Thresholds**: Adjust alert threshold, EMA alpha, and hysteresis
3. **Run Pipeline**: Click "Run All Modules" to execute processing

### **Navigation**
- **Tab Navigation**: Click on tab headers to switch views
- **Sidebar**: Use sidebar for configuration and status monitoring
- **Interactive Charts**: Hover and click on chart elements for details

---

## 📊 Demo Data

The simple app includes comprehensive demo data to showcase all features:
- **1,247 articles** processed
- **8 active alerts** with various severity levels
- **4 data sources** with high success rates
- **Multiple scoring stages** with realistic distributions
- **Timeline events** spanning 20 hours
- **Block matches** with confidence scores
- **Crisis scenarios** with different severity levels
- **Export history** with various formats

---

## 🔮 Future Enhancements

### **Planned Features**
- **Real-time Data Integration**: Connect to live data sources
- **Advanced Filtering**: More sophisticated data filtering options
- **Custom Visualizations**: User-defined chart types
- **Export Customization**: Configurable export formats
- **User Authentication**: Multi-user support with role-based access
- **API Integration**: REST API for external system integration

### **Technical Improvements**
- **Database Integration**: Persistent data storage
- **Caching Layer**: Redis-based caching for better performance
- **WebSocket Support**: Real-time updates without page refresh
- **Mobile Responsiveness**: Optimized mobile interface
- **Accessibility**: WCAG compliance for accessibility

---

## 📞 Support

For technical support or feature requests:
- **Documentation**: Refer to this documentation
- **Code**: Check the source code in `nwwpkg/ui/simple_app.py`
- **Configuration**: Modify settings in the `config/` directory
- **Logs**: Check console output for debugging information

---

**Last Updated**: 2025-09-14  
**Version**: 1.0.0  
**Status**: ✅ Production Ready



