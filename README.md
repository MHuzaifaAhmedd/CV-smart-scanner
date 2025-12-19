# Smart CV Optimizer 📄

An intelligent CV optimization tool that helps job seekers pass Applicant Tracking Systems (ATS) by analyzing CV-Job Description alignment using AI.

## 🎯 Features

- **ATS Match Scoring**: Get a 0-100% match score based on:
  - Technical skills match (50%)
  - Years of experience alignment (30%)
  - Keyword coverage (20%)

- **Keyword Gap Analysis**: Identify missing/weak keywords grouped by:
  - Required Skills
  - Preferred Skills
  - Tools & Technologies
  - Soft Skills

- **Bullet Point Optimization**: Get 3-5 CV bullets rewritten using the X-Y-Z formula:
  - "Accomplished [X] as measured by [Y], by doing [Z]"

- **ATS Red Flag Detection**: Identify formatting issues that ATS systems struggle with:
  - Multi-column layouts
  - Tables and graphics
  - Non-standard section headers
  - Icons and special formatting

- **🆕 CV Revision & DOCX Export**: Generate a professionally revised CV:
  - Incorporates missing keywords from analysis
  - Improves wording, grammar, and impact statements
  - Preserves original structure, headings, and bullet points
  - Downloads as ATS-friendly Word document (.docx)
  - Text preview with formatted sections

## 🚀 Quick Start

### 1. Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

*Note: Using a virtual environment prevents conflicts with system packages and keeps dependencies isolated.*

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys

Create a `.env` file in the project root:

```bash
# OpenAI API Key (for GPT-4o) - REQUIRED
OPENAI_API_KEY=sk-your-openai-api-key-here

# Google Gemini API Key (optional, for Gemini 1.5 Pro)
GOOGLE_API_KEY=your-google-api-key-here
```

**Get API Keys:**
- OpenAI: https://platform.openai.com/api-keys
- Google Gemini: https://makersuite.google.com/app/apikey

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 How to Use

1. **Upload CV**: Click "Browse files" and upload your CV in PDF format
2. **Paste Job Description**: Copy the complete job description and paste it in the text area
3. **Select LLM**: Choose between GPT-4o (default) or Gemini 1.5 Pro in the sidebar
4. **Analyze**: Click the "🚀 Analyze CV" button
5. **Review Results**: Get instant feedback on match score, keyword gaps, and optimization suggestions
6. **🆕 Generate Revised CV** (Optional): Expand the "✏️ Generate Revised CV" section and click the button to:
   - Generate a professionally revised CV incorporating all feedback
   - Preview the revised content with formatted sections
   - Download as ATS-friendly Word document (.docx)

## 🎨 UI Layout

```
┌─────────────────────────────────────────────┐
│         Smart CV Optimizer                  │
├──────────────────┬──────────────────────────┤
│   INPUT          │   ANALYSIS RESULTS       │
│                  │                          │
│ 📥 Upload CV     │ 📊 Match Score: 85%     │
│ 📝 Paste JD      │ ━━━━━━━━━━━━━━━━━━━━━━  │
│ 🚀 Analyze       │ 📊 Score Breakdown       │
│                  │ 🔍 Keyword Gaps          │
│                  │ ✍️  Bullet Optimizations │
│                  │ 🚨 ATS Red Flags         │
└──────────────────┴──────────────────────────┘
```

## 🔧 Configuration

### Token Limits (Cost Control)
- CV Text: Max 12,000 characters (~3,000 tokens)
- Job Description: Max 6,000 characters (~1,500 tokens)
- Texts longer than limits are automatically truncated

### Retry Logic
- Max retries: 3 attempts
- Exponential backoff: 1s, 2s, 4s delays
- Timeout: 60 seconds per API call

### Caching
- PDF extraction: Cached based on file content (Streamlit cache)
- LLM responses: Cached in session state based on input hash
- Prevents duplicate API calls and speeds up reruns

## 🏗️ Technical Architecture

### Core Components

```
app.py (600+ lines)
├── PDF Text Extraction (pdfplumber + caching)
├── Text Processing (truncation, normalization)
├── LLM Integration
│   ├── OpenAI GPT-4o
│   └── Google Gemini 1.5 Pro
├── Score Validation (clamping to valid ranges)
├── UI Rendering
│   ├── Score Breakdown
│   ├── Keyword Gaps
│   ├── Bullet Optimizations
│   └── ATS Red Flags
└── Error Handling (retries, timeouts, validation)
```

### Key Functions

#### Analysis Functions
1. `extract_text_from_pdf()` - PDF extraction with caching
2. `truncate_text()` - Token control for cost optimization
3. `call_llm_api()` - Main LLM wrapper with caching
4. `validate_and_clamp_scores()` - Score validation (0-100 range)
5. `render_analysis_results()` - Complete UI rendering
6. `render_score_breakdown()` - Visual score component
7. `render_keyword_gaps()` - Keyword analysis section
8. `render_ats_red_flags()` - Red flag detection section

#### 🆕 CV Revision Functions
9. `create_revision_prompt()` - LLM prompt for structure-preserving revision
10. `call_revision_llm()` - Main revision wrapper with caching
11. `call_openai_revision_api()` - GPT-4o revision API call
12. `call_gemini_revision_api()` - Gemini revision API call
13. `validate_revised_cv()` - Output validation with fallback logic
14. `parse_cv_sections()` - Parse CV into structured sections
15. `generate_docx_cv()` - Create ATS-friendly Word document
16. `render_cv_revision_section()` - UI for revision generation and download

## 🛡️ Production Features

### Reliability & Safety
- ✅ Score clamping (ensures 0-100 range)
- ✅ Hallucination prevention (strict prompts)
- ✅ Retry logic with exponential backoff
- ✅ Timeout handling (60s per call)
- ✅ Defensive rendering (partial data support)

### Performance & Cost
- ✅ Token control (12k CV / 6k JD limits)
- ✅ PDF extraction caching
- ✅ LLM response caching
- ✅ Session state optimization

### ATS Accuracy
- ✅ Keyword frequency emphasis
- ✅ Section header validation
- ✅ Formatting issue detection
- ✅ Strict keyword extraction (JD only)

## 📝 Dependencies

- `streamlit>=1.32.0` - Web UI framework
- `pdfplumber>=0.10.0` - PDF text extraction
- `openai>=1.12.0` - GPT-4o API
- `google-generativeai>=0.4.0` - Gemini API
- `python-dotenv>=1.0.0` - Environment variables
- `python-docx>=0.8.11` - Word document generation (for CV revision feature)

## 🧪 Testing Checklist

### Analysis Testing
- [ ] Upload a PDF CV
- [ ] Paste a job description (minimum 50 characters)
- [ ] Verify match score appears (0-100%)
- [ ] Check score breakdown shows three components
- [ ] Review keyword gaps by category
- [ ] Inspect bullet point optimizations
- [ ] Check ATS red flags and fixes
- [ ] Test with both GPT-4o and Gemini (if both API keys configured)
- [ ] Verify truncation warnings for large inputs
- [ ] Test caching by re-clicking "Analyze" without changes

### 🆕 CV Revision Testing
- [ ] Expand "✏️ Generate Revised CV" section after analysis
- [ ] Click "🚀 Generate Revised CV" button
- [ ] Verify revised CV appears in preview section
- [ ] Check that original structure is preserved (same headings, bullets)
- [ ] Verify missing keywords are integrated naturally
- [ ] Check grammar and clarity improvements
- [ ] Click "📥 Download as Word Document" button
- [ ] Open downloaded .docx file and verify formatting
- [ ] Test with both GPT-4o and Gemini models
- [ ] Verify performance warning appears for large CVs (> 15k chars)
- [ ] Test caching by regenerating without changes

## 🐛 Troubleshooting

### "OpenAI API key not found"
- Ensure `.env` file exists in project root
- Check variable name: `OPENAI_API_KEY=sk-...`
- Restart Streamlit after adding `.env`

### "Failed to extract text from PDF"
- Ensure PDF is not password-protected
- Check PDF is not scanned image (needs OCR)
- Try re-saving PDF from original source

### "OpenAI API Quota Exceeded" (Error 429)

**What it means:** You've hit your OpenAI API usage limit or haven't set up billing.

**⚠️ Common Issue: Usage shows $0.00 but still getting quota error?**
- **This almost always means:** No payment method is added to your account
- **Even with $0 usage**, OpenAI requires a payment method on file to use the API
- **Solution:** Add a payment method (see below)

**How to check your quota:**
1. Visit: https://platform.openai.com/usage
2. Check billing: https://platform.openai.com/account/billing
3. View limits: https://platform.openai.com/account/limits

**Solutions (in order of priority):**

1. **Add Payment Method (REQUIRED):**
   - 👉 https://platform.openai.com/account/billing/payment-methods
   - Add a credit/debit card
   - **Even if you have free credits, payment method is required**
   - Set spending limits if you want to control costs: https://platform.openai.com/account/billing/limits

2. **Check Account Status:**
   - Verify billing is active: https://platform.openai.com/account/billing
   - Ensure no account restrictions

3. **Alternative: Use Gemini (No Payment Required):**
   - Switch to "Gemini 1.5 Pro" in sidebar
   - Get free API key: https://makersuite.google.com/app/apikey
   - Free tier available without payment method

### "Gemini Model Not Found" (Error 404)

**What it means:** The Gemini model name isn't recognized by your API version/region.

**Solutions:**
1. **The app auto-tries multiple model names** - it should work automatically
2. **Check your API key:**
   - Ensure it's from: https://makersuite.google.com/app/apikey
   - Make sure it's a valid Gemini API key (not Vertex AI)
3. **Check available models:**
   - Visit: https://ai.google.dev/models/gemini
   - The app will automatically list and use available models
4. **Region/API Version:**
   - Different regions may have different model names
   - The app handles this automatically by trying multiple variants
5. **Update API key:**
   - Get a fresh key: https://makersuite.google.com/app/apikey
   - Ensure you're using the Generative AI API (not Vertex AI)

**Note:** The app tries these model names automatically:
- `gemini-1.5-pro`
- `gemini-1.5-flash`
- `gemini-pro`
- Plus variants with `models/` prefix

### "Failed to parse valid JSON after 3 attempts"
- Check API key has sufficient credits
- Verify internet connection
- Try alternative LLM (GPT-4o vs Gemini)

### High API Costs
- Use token limits (already set to 12k CV / 6k JD)
- Leverage caching (don't re-analyze unnecessarily)
- Consider using Gemini (often more cost-effective)

## 📊 Example Results

```
Match Score: 78%
├─ Technical Skills: 38/50 (76%)
├─ Experience Alignment: 24/30 (80%)
└─ Keyword Coverage: 16/20 (80%)

Keyword Gaps:
├─ Required Skills: Python, Docker, Kubernetes
├─ Tools & Technologies: Jenkins, AWS Lambda
└─ Soft Skills: Leadership, Stakeholder Management

ATS Red Flags:
├─ Multi-column layout detected
└─ Non-standard section header: "Professional Journey"
```

## 🤝 Contributing

This is a production-ready implementation following best practices:
- Clean, modular code with clear separation of concerns
- Comprehensive error handling and validation
- Detailed comments explaining design decisions
- Defensive programming for robust operation

## 📄 License

MIT License - Feel free to use and modify for your needs.

---

Built with ❤️ using Streamlit | Powered by GPT-4o & Gemini 1.5 Pro

