# NWW Dashboard UI Structure Diagram

## 🏗️ Overall Layout Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           🌍 NWW Dashboard Header                          │
│                    Complete Automation Package for Crisis Detection        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┬───────────────────────────────────────────────────────────┐
│                 │                                                           │
│   SIDEBAR       │                    MAIN CONTENT AREA                     │
│                 │                                                           │
│ ⚙️ Configuration│  ┌─────────────────────────────────────────────────────┐  │
│                 │  │                TAB NAVIGATION                      │  │
│ Bundle Directory│  │ [📊 Overview] [📥 Ingest] [🎯 Scoring] [⏰ Timeline] │  │
│                 │  │ [🧱 Blocks] [📋 Scenarios] [📦 Artifacts] [📝 Ledger] │  │
│ 🔧 Settings     │  └─────────────────────────────────────────────────────┘  │
│                 │                                                           │
│ Alert Threshold │  ┌─────────────────────────────────────────────────────┐  │
│ EMA Alpha       │  │                                                     │  │
│ Hysteresis      │  │                TAB CONTENT                          │  │
│                 │  │                                                     │  │
│ 🚀 Run All      │  │  (Dynamic content based on selected tab)            │  │
│                 │  │                                                     │  │
│ 📈 Status       │  │  • Charts and visualizations                        │  │
│ ✅ Ingest       │  │  • Data tables                                      │  │
│ ✅ Normalize    │  │  • Control buttons                                  │  │
│ ✅ Analyze      │  │  • Metrics and KPIs                                 │  │
│ ✅ Gate         │  │  • Interactive elements                             │  │
│ ✅ Score IS     │  │                                                     │  │
│ ✅ Score DBN    │  │                                                     │  │
│ ✅ Score LLM    │  │                                                     │  │
│ ✅ Fusion       │  │                                                     │  │
│ ✅ Blocks       │  │                                                     │  │
│ ✅ Scenarios    │  │                                                     │  │
│ ✅ Alerts       │  │                                                     │  │
│ ✅ Ledger       │  │                                                     │  │
│                 │  │                                                     │  │
└─────────────────┴───────────────────────────────────────────────────────────┘
```

## 📊 Tab 1: Overview Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              📊 Overview Tab                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ Articles    │ │ Active      │ │ Avg Score   │ │ System      │          │
│  │ Processed   │ │ Alerts      │ │             │ │ Health      │          │
│  │ 1,247 ↗️ 12 │ │ 8 ↗️ 3      │ │ 0.65 ↗️ 0.05│ │ 🟢 99% ↗️   │          │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘          │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        🔄 Processing Pipeline                          │ │
│  │                                                                         │ │
│  │ ✅ Ingest: Extract articles from sources                               │ │
│  │ ✅ Normalize: Clean and normalize text                                 │ │
│  │ ✅ Analyze: Extract keywords, entities, frames                         │ │
│  │ ✅ Gate: Filter content based on indicators                            │ │
│  │ ✅ Score IS: Indicator-based scoring                                   │ │
│  │ ✅ Score DBN: Dynamic Bayesian network scoring                         │ │
│  │ ✅ Score LLM: LLM judge scoring                                        │ │
│  │ ✅ Fusion: Score fusion and calibration                                │ │
│  │ ✅ Blocks: EDS block matching                                          │ │
│  │ ✅ Scenarios: Scenario construction                                    │ │
│  │ ✅ Alerts: Alert generation                                            │ │
│  │ ✅ Ledger: Audit trail                                                 │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        📈 Recent Activity                              │ │
│  │                                                                         │ │
│  │  [Interactive Plotly Chart - Dual Y-axis]                              │ │
│  │  • Blue line: Articles processed over time                             │ │
│  │  • Red line: Alerts generated over time                                │ │
│  │  • 24-hour timeline with hourly data points                            │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📥 Tab 2: Ingest Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              📥 Ingest Tab                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────┐ ┌─────────────────────────────────┐ │
│  │            📋 Sources               │ │         ⚙️ Controls             │ │
│  │                                     │ │                                 │ │
│  │ ┌─────────────────────────────────┐ │ │ ┌─────────────────────────────┐ │ │
│  │ │ URL                │ Status     │ │ │ │ 🔄 Refresh Sources          │ │ │
│  │ │ Last Update        │ Success    │ │ │ └─────────────────────────────┘ │ │
│  │ │ ───────────────────────────────── │ │                                 │ │
│  │ │ reuters.com/world  │ Active     │ │ │ ┌─────────────────────────────┐ │ │
│  │ │ 2 min ago         │ 98%        │ │ │ │ ▶️ Run Ingest               │ │ │
│  │ │ ───────────────────────────────── │ │ └─────────────────────────────┘ │ │
│  │ │ bbc.com/news       │ Active     │ │                                 │ │
│  │ │ 5 min ago         │ 95%        │ │                                 │ │
│  │ │ ───────────────────────────────── │ │                                 │ │
│  │ │ cnn.com/world      │ Active     │ │                                 │ │
│  │ │ 1 min ago         │ 92%        │ │                                 │ │
│  │ │ ───────────────────────────────── │ │                                 │ │
│  │ │ ap.org/news        │ Active     │ │                                 │ │
│  │ │ 3 min ago         │ 97%        │ │                                 │ │
│  │ └─────────────────────────────────┘ │                                 │ │
│  └─────────────────────────────────────┘ └─────────────────────────────────┘ │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                            📊 Statistics                               │ │
│  │                                                                         │ │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                        │ │
│  │ │ Total       │ │ Success     │ │ Last        │                        │ │
│  │ │ Sources     │ │ Rate        │ │ Update      │                        │ │
│  │ │ 4           │ │ 95% ↗️ 2%   │ │ 2 min ago   │                        │ │
│  │ └─────────────┘ └─────────────┘ └─────────────┘                        │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🎯 Tab 3: Scoring Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              🎯 Scoring Tab                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        📊 Score Distribution                           │ │
│  │                                                                         │ │
│  │  [Interactive Histogram Chart]                                         │ │
│  │  • 20 bins showing score frequency distribution                        │ │
│  │  • X-axis: Score values (0.0 - 1.0)                                   │ │
│  │  • Y-axis: Number of articles                                          │ │
│  │  • Hover effects for detailed information                              │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        📈 Scores by Stage                              │ │
│  │                                                                         │ │
│  │ ┌─────────┬─────────┬─────────┬─────────┐                              │ │
│  │ │ Stage   │ Mean    │ Std     │ Count   │                              │ │
│  │ ├─────────┼─────────┼─────────┼─────────┤                              │ │
│  │ │ IS      │ 0.65    │ 0.12    │ 1247    │                              │ │
│  │ │ DBN     │ 0.68    │ 0.15    │ 1247    │                              │ │
│  │ │ LLM     │ 0.62    │ 0.18    │ 1247    │                              │ │
│  │ │ FUSION  │ 0.67    │ 0.14    │ 1247    │                              │ │
│  │ └─────────┴─────────┴─────────┴─────────┘                              │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        🏆 Top Scoring Articles                         │ │
│  │                                                                         │ │
│  │ ┌─────┬─────────┬─────────┐                                            │ │
│  │ │ ID  │ Stage   │ Score   │                                            │ │
│  │ ├─────┼─────────┼─────────┤                                            │ │
│  │ │ a1  │ FUSION  │ 0.95    │                                            │ │
│  │ │ a2  │ FUSION  │ 0.89    │                                            │ │
│  │ │ a3  │ FUSION  │ 0.87    │                                            │ │
│  │ │ a4  │ FUSION  │ 0.84    │                                            │ │
│  │ │ a5  │ FUSION  │ 0.82    │                                            │ │
│  │ └─────┴─────────┴─────────┘                                            │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## ⏰ Tab 4: Timeline Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ⏰ Timeline Tab                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        📅 Event Timeline                               │ │
│  │                                                                         │ │
│  │  [Interactive Timeline Chart - Gantt Style]                            │ │
│  │                                                                         │ │
│  │  Military     ████████████████████████████████████████████████████████  │ │
│  │  Diplomatic   ████████████████████████████████████████████████████████  │ │
│  │  Economic     ████████████████████████████████████████████████████████  │ │
│  │  Military     ████████████████████████████████████████████████████████  │ │
│  │  Diplomatic   ████████████████████████████████████████████████████████  │ │
│  │                                                                         │ │
│  │  • X-axis: Time range (start and end times)                            │ │
│  │  • Y-axis: Event types                                                  │ │
│  │  • Color coding: Severity levels (High, Medium, Low, Critical)         │ │
│  │  • 10 sample events over 20-hour period                                │ │
│  │  • Hover effects for event details                                      │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🧱 Tab 5: Blocks Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              🧱 Blocks Tab                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        EDS Block Analysis                              │ │
│  │                                                                         │ │
│  │ ┌─────────┬─────────────┬─────────────┬─────────────┬─────────────┐    │ │
│  │ │ Block   │ Type        │ Confidence  │ Articles    │ Status      │    │ │
│  │ │ ID      │             │             │             │             │    │ │
│  │ ├─────────┼─────────────┼─────────────┼─────────────┼─────────────┤    │ │
│  │ │ B001    │ Military    │ 0.95        │ 5           │ Active      │    │ │
│  │ │ B002    │ Diplomatic  │ 0.87        │ 3           │ Active      │    │ │
│  │ │ B003    │ Economic    │ 0.76        │ 7           │ Resolved    │    │ │
│  │ │ B004    │ Military    │ 0.92        │ 4           │ Active      │    │ │
│  │ └─────────┴─────────────┴─────────────┴─────────────┴─────────────┘    │ │
│  │                                                                         │ │
│  │ • Block matching results from EDS processing                           │ │
│  │ • Confidence scores range from 0.76 to 0.95                           │ │
│  │ • Status tracking (Active, Resolved)                                   │ │
│  │ • Article count per block                                              │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📋 Tab 6: Scenarios Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              📋 Scenarios Tab                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        Scenario Construction Results                    │ │
│  │                                                                         │ │
│  │ ┌─────────┬─────────────────┬─────────────┬─────────────┬─────────────┐ │ │
│  │ │ Scenario│ Name            │ Severity    │ Confidence  │ Status      │ │ │
│  │ │ ID      │                 │             │             │             │ │ │
│  │ ├─────────┼─────────────────┼─────────────┼─────────────┼─────────────┤ │ │
│  │ │ S001    │ Military        │ High        │ 0.89        │ Active      │ │ │
│  │ │         │ Conflict        │             │             │             │ │ │
│  │ ├─────────┼─────────────────┼─────────────┼─────────────┼─────────────┤ │ │
│  │ │ S002    │ Diplomatic      │ Medium      │ 0.76        │ Monitoring  │ │ │
│  │ │         │ Crisis          │             │             │             │ │ │
│  │ ├─────────┼─────────────────┼─────────────┼─────────────┼─────────────┤ │ │
│  │ │ S003    │ Economic        │ Medium      │ 0.68        │ Resolved    │ │ │
│  │ │         │ Warfare         │             │             │             │ │ │
│  │ └─────────┴─────────────────┴─────────────┴─────────────┴─────────────┘ │ │
│  │                                                                         │ │
│  │ • Crisis scenario construction results                                  │ │
│  │ • Severity levels: High, Medium, Low                                   │ │
│  │ • Confidence scores for scenario accuracy                              │ │
│  │ • Status tracking: Active, Monitoring, Resolved                        │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📦 Tab 7: Artifacts Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              📦 Artifacts Tab                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                            📤 Export Options                           │ │
│  │                                                                         │ │
│  │ ┌─────────────────────────┐ ┌─────────────────────────────────────────┐ │ │
│  │ │ 📄 Export JSON          │ │ 📦 Export ZIP                           │ │ │
│  │ └─────────────────────────┘ └─────────────────────────────────────────┘ │ │
│  │                                                                         │ │
│  │ ┌─────────────────────────┐ ┌─────────────────────────────────────────┐ │ │
│  │ │ 📋 Export Brief         │ │ 🔒 Export STIX                          │ │ │
│  │ └─────────────────────────┘ └─────────────────────────────────────────┘ │ │
│  │                                                                         │ │
│  │ • JSON: Structured data export                                         │ │
│  │ • Brief: Human-readable summary report                                 │ │
│  │ • ZIP: Compressed archive with all data                                │ │
│  │ • STIX: Threat intelligence format for international cooperation       │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                            📚 Export History                           │ │
│  │                                                                         │ │
│  │ ┌─────────────┬─────────────┬─────────────┬─────────────┐              │ │
│  │ │ Timestamp   │ Format      │ Size        │ Status      │              │ │
│  │ ├─────────────┼─────────────┼─────────────┼─────────────┤              │ │
│  │ │ 2025-09-14  │ JSON        │ 2.3 MB      │ Completed   │              │ │
│  │ │ 10:30       │             │             │             │              │ │
│  │ ├─────────────┼─────────────┼─────────────┼─────────────┤              │ │
│  │ │ 2025-09-14  │ STIX        │ 1.8 MB      │ Completed   │              │ │
│  │ │ 09:15       │             │             │             │              │ │
│  │ ├─────────────┼─────────────┼─────────────┼─────────────┤              │ │
│  │ │ 2025-09-14  │ ZIP         │ 4.1 MB      │ Completed   │              │ │
│  │ │ 08:00       │             │             │             │              │ │
│  │ └─────────────┴─────────────┴─────────────┴─────────────┘              │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📝 Tab 8: Ledger Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              📝 Ledger Tab                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                            Audit Trail                                 │ │
│  │                                                                         │ │
│  │ ┌─────────────────────┬─────────────┬─────────────────────────────────┐ │ │
│  │ │ Timestamp           │ Step        │ Description                     │ │ │
│  │ ├─────────────────────┼─────────────┼─────────────────────────────────┤ │ │
│  │ │ 2025-09-14          │ Alerts      │ Alert generation completed      │ │ │
│  │ │ 10:30:00            │             │                                 │ │ │
│  │ ├─────────────────────┼─────────────┼─────────────────────────────────┤ │ │
│  │ │ 2025-09-14          │ Scenarios   │ Scenario construction completed │ │ │
│  │ │ 10:25:00            │             │                                 │ │ │
│  │ ├─────────────────────┼─────────────┼─────────────────────────────────┤ │ │
│  │ │ 2025-09-14          │ Blocks      │ Block matching completed        │ │ │
│  │ │ 10:20:00            │             │                                 │ │ │
│  │ └─────────────────────┴─────────────┴─────────────────────────────────┘ │ │
│  │                                                                         │ │
│  │ • Complete audit trail of all processing steps                         │ │
│  │ • Precise timestamps to the second                                     │ │
│  │ • Status tracking for compliance and debugging                         │ │
│  │ • Chronological order of operations                                    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🎨 Visual Design Elements

### **Color Coding System**
```
🟢 Green    - Success, Completed, Healthy
🔴 Red      - Alerts, Warnings, Critical
🔵 Blue     - Primary data, Information
🟠 Orange   - Secondary metrics, Warnings
🟣 Purple   - Special information, Highlights
⚫ Black     - Text, Borders, Default
```

### **Icon System**
```
📊 Overview    - Dashboard and metrics
📥 Ingest      - Data collection
🎯 Scoring     - Analysis and scoring
⏰ Timeline    - Temporal analysis
🧱 Blocks      - Block processing
📋 Scenarios   - Scenario construction
📦 Artifacts   - Export and artifacts
📝 Ledger      - Audit and logging
⚙️ Settings    - Configuration
🚀 Actions     - Execution buttons
📈 Status      - Progress indicators
✅ Success     - Completed operations
❌ Error       - Failed operations
🔄 Refresh     - Update operations
▶️ Run         - Start operations
```

### **Layout Principles**
- **Responsive Design**: Adapts to different screen sizes
- **Consistent Spacing**: Uniform margins and padding
- **Clear Hierarchy**: Visual hierarchy through typography and spacing
- **Interactive Elements**: Hover effects and click feedback
- **Accessibility**: High contrast and readable fonts
- **Professional Appearance**: Clean, modern interface design



