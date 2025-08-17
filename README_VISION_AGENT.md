# Vision Analysis Agent - Documentation

## Overview
The Vision Analysis Agent has been enhanced to support both demo mode and real Gemini Vision API mode for analyzing plant and soil images.

## Features

### 🎭 Demo Mode
- **Purpose**: Test the system without requiring a Gemini API key
- **Functionality**: Generates realistic fake predictions and analysis
- **Use Case**: Development, testing, and demonstrations
- **Requirements**: No API key needed

### 🔬 Real Mode
- **Purpose**: Actual image analysis using Google's Gemini Vision API
- **Functionality**: Analyzes real images for plant diseases, soil conditions, etc.
- **Use Case**: Production use with real agricultural analysis
- **Requirements**: Valid `GEMINI_API_KEY` environment variable

## Configuration

### Environment Variables
```bash
# Set your Gemini API key for real mode
export GEMINI_API_KEY="your_api_key_here"
```

### API Key Setup
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Set the environment variable:
   ```bash
   export GEMINI_API_KEY="your_key_here"
   ```

## Usage

### Frontend Interface
1. **Upload Image**: Select a plant or soil image
2. **Choose Mode**: Select "plant" or "soil" analysis
3. **Demo Mode**: Check/uncheck the "Demo Mode" checkbox
   - ✅ Checked = Demo mode (fake predictions)
   - ❌ Unchecked = Real mode (Gemini Vision API)
4. **Analyze**: Click "Analyze Image" button

### Backend API
```python
# Demo mode
POST /analyze-image
{
    "file": <image_file>,
    "mode": "plant",
    "demo_mode": "true"
}

# Real mode
POST /analyze-image
{
    "file": <image_file>,
    "mode": "plant", 
    "demo_mode": "false"
}
```

## File Changes

### 1. `agents.py`
- **VisionAnalysisAgent class**: Enhanced with demo/real mode logic
- **Image format detection**: Automatically detects JPEG, PNG, WebP, GIF
- **Debug logging**: Comprehensive request/response logging
- **Error handling**: Better error messages and fallbacks

### 2. `backend.py`
- **Endpoint**: `/analyze-image` now accepts `demo_mode` parameter
- **Form handling**: Processes checkbox state from frontend
- **Parameter passing**: Forwards demo mode to orchestrator

### 3. `routing.py`
- **SimpleAgriculturalAI**: Routes image analysis through vision agent
- **Demo mode**: Sets agent's demo mode before analysis
- **Debug logging**: Logs analysis requests and parameters

### 4. `static/index.html`
- **UI**: Added demo mode checkbox (unchecked by default)
- **JavaScript**: Sends checkbox state to backend
- **User experience**: Clear indication of mode selection

## Testing

### Test Script
Run `test_vision_agent.py` to test both modes:
```bash
python test_vision_agent.py
```

### Manual Testing
1. **Demo Mode**: Uncheck demo mode, upload image → Should get fake prediction
2. **Real Mode**: Check demo mode, upload image → Should get Gemini analysis (if API key set)

## Troubleshooting

### Common Issues

#### 1. "Gemini API key not configured"
- **Cause**: `GEMINI_API_KEY` environment variable not set
- **Solution**: Set the environment variable or use demo mode

#### 2. "400 Bad Request" Error
- **Cause**: API request format issue (usually fixed now)
- **Solution**: Check debug logs for request details

#### 3. "Vision analysis failed"
- **Cause**: Network issues, API limits, or malformed requests
- **Solution**: Check internet connection and API key validity

### Debug Information
The agent logs detailed information:
- Request URL and parameters
- Image format and size
- API response status and headers
- Error details for failed requests

## Performance

### Demo Mode
- **Speed**: Instant response (no API calls)
- **Cost**: Free
- **Quality**: Realistic but fake data

### Real Mode
- **Speed**: 2-5 seconds (API call dependent)
- **Cost**: Based on Gemini API pricing
- **Quality**: Real AI analysis

## Future Enhancements

### Planned Features
1. **Batch Processing**: Analyze multiple images at once
2. **Result Caching**: Cache analysis results for repeated images
3. **Advanced Prompts**: More specialized analysis modes
4. **Export Options**: PDF reports, CSV data export

### Integration Ideas
1. **Database Storage**: Save analysis results
2. **User Management**: Track analysis history
3. **Mobile App**: Native mobile interface
4. **API Rate Limiting**: Prevent abuse

## Support

### Getting Help
1. Check debug logs for error details
2. Verify environment variables
3. Test with demo mode first
4. Check API key validity

### Reporting Issues
Include:
- Error message
- Debug log output
- Image format and size
- Demo vs real mode used
- Environment details

---

**Note**: This documentation reflects the current state of the Vision Analysis Agent. For the latest updates, check the code comments and commit history.
