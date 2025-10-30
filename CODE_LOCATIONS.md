# 📍 Exact Code Locations for All Features

## 1. Dark Mode Toggle

### Implementation Files:
- **`frontend/src/App.jsx`** (Lines 13-34)
- **`frontend/src/components/Navbar.jsx`** (Lines 5, 96-108)

### Code Snippets:

**App.jsx - State Management:**
```javascript
// Line 13
const [darkMode, setDarkMode] = useState(false);

// Lines 16-23 - Load saved preference
useEffect(() => {
  const savedDarkMode = localStorage.getItem('darkMode') === 'true';
  setDarkMode(savedDarkMode);
  if (savedDarkMode) {
    document.documentElement.classList.add('dark');
  }
}, []);

// Lines 25-34 - Toggle function
const toggleDarkMode = () => {
  const newDarkMode = !darkMode;
  setDarkMode(newDarkMode);
  localStorage.setItem('darkMode', newDarkMode.toString());
  if (newDarkMode) {
    document.documentElement.classList.add('dark');
  } else {
    document.documentElement.classList.remove('dark');
  }
};
```

**Navbar.jsx - Button:**
```javascript
// Lines 96-108
<motion.button
  whileHover={{ scale: 1.1, rotate: 180 }}
  whileTap={{ scale: 0.9 }}
  onClick={toggleDarkMode}
  className="ml-4 p-2 rounded-xl bg-gray-100 dark:bg-gray-800"
>
  {darkMode ? (
    <Sun className="w-5 h-5 text-yellow-500" />
  ) : (
    <Moon className="w-5 h-5 text-gray-700" />
  )}
</motion.button>
```

---

## 2. Grad-CAM Heatmap

### Backend Implementation:
- **`backend/server.js`** (Lines 54-83, 86-115, 167-200)

### Backend Code:

**Enable Visualization Flag:**
```javascript
// Lines 54-62
const pythonProcess = spawn('python', [
  pythonScript,
  'inference',
  '--config', configPath,
  '--checkpoint', checkpointPath,
  '--image', imagePath,
  '--device', 'auto',
  '--visualize'  // ← THIS ENABLES GRAD-CAM
]);
```

**Parse Grad-CAM Path:**
```javascript
// Lines 86-115
function parsePythonOutput(output) {
  const result = {
    prediction: 'Unknown',
    probability: 0,
    confidence: 0,
    gradcamPath: null  // ← STORES PATH
  };
  
  lines.forEach(line => {
    // ... other parsing ...
    if (line.includes('Visualization saved:')) {
      const match = line.match(/Visualization saved:\s*(.+)/);
      if (match) result.gradcamPath = match[1].trim();
    }
  });
  
  return result;
}
```

**Read and Return Base64:**
```javascript
// Lines 172-200
// Read Grad-CAM image if generated
let gradcamBase64 = null;
if (result.gradcamPath) {
  try {
    const gradcamData = await fs.readFile(result.gradcamPath);
    gradcamBase64 = `data:image/png;base64,${gradcamData.toString('base64')}`;
    await fs.unlink(result.gradcamPath);
  } catch (error) {
    console.error('Error reading Grad-CAM:', error);
  }
}

res.json({
  success: true,
  result: {
    prediction: result.prediction,
    probability: result.probability,
    confidence: result.confidence,
    gradcamImage: gradcamBase64,  // ← RETURNED TO FRONTEND
    message: result.prediction === 'Cancer' 
      ? 'Cancer detected - Please consult a medical professional immediately' 
      : 'No cancer detected - Routine follow-up recommended'
  }
});
```

### Frontend Implementation:
- **`frontend/src/components/ResultsSection.jsx`** (Lines 8, 21, 262-326)
- **`frontend/src/App.jsx`** (Lines 14, 54-59, 74-78)

### Frontend Code:

**Store Uploaded Image:**
```javascript
// App.jsx Lines 54-59
const imageBase64 = await new Promise((resolve) => {
  const reader = new FileReader();
  reader.onloadend = () => resolve(reader.result);
  reader.readAsDataURL(file);
});
setUploadedImage(imageBase64);
```

**Add to Result:**
```javascript
// App.jsx Lines 74-78
const resultWithImage = {
  ...data.result,
  originalImage: imageBase64  // ← ORIGINAL IMAGE
};
setResult(resultWithImage);
```

**Grad-CAM Section:**
```javascript
// ResultsSection.jsx Lines 262-326
{hasGradcam && (
  <motion.div className="...">
    {/* Header */}
    <div className="flex items-center justify-between mb-6">
      <div>
        <h3>AI Explainability</h3>
        <p>Grad-CAM heatmap showing regions the AI focused on</p>
      </div>
      
      {/* Toggle Button */}
      <motion.button
        onClick={() => setShowHeatmap(!showHeatmap)}
        className="..."
      >
        {showHeatmap ? (
          <><EyeOff /><span>Original</span></>
        ) : (
          <><Eye /><span>Heatmap</span></>
        )}
      </motion.button>
    </div>

    {/* Animated Image */}
    <AnimatePresence mode="wait">
      <motion.div
        key={showHeatmap ? 'heatmap' : 'original'}
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        transition={{ duration: 0.3 }}
      >
        <img
          src={showHeatmap ? result.gradcamImage : result.originalImage}
          alt={showHeatmap ? 'Grad-CAM Heatmap' : 'Original X-ray'}
        />
        <div className="badge">
          {showHeatmap ? '🔥 Heatmap View' : '📷 Original View'}
        </div>
      </motion.div>
    </AnimatePresence>
  </motion.div>
)}
```

---

## 3. Result Card Glow Effects

### Implementation File:
- **`frontend/src/components/ResultsSection.jsx`** (Lines 114-174)

### Code:

**Conditional Styling:**
```javascript
// Lines 118-123
className={`relative overflow-hidden rounded-3xl shadow-2xl p-8 ${
  isCancer
    ? 'bg-gradient-to-br from-red-50 to-pink-50 dark:from-red-900/20 dark:to-pink-900/20 border-2 border-red-200 dark:border-red-800'
    : 'bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 border-2 border-green-200 dark:border-green-800'
}`}
```

**Pulsing Glow Animation:**
```javascript
// Lines 125-134
<motion.div
  animate={{
    scale: [1, 1.2, 1],
    opacity: [0.1, 0.2, 0.1],
  }}
  transition={{ duration: 3, repeat: Infinity }}
  className={`absolute inset-0 ${
    isCancer ? 'bg-red-500' : 'bg-green-500'
  } blur-3xl`}
/>
```

**Icon and Text:**
```javascript
// Lines 146-172
{isCancer ? (
  <AlertCircle className="w-10 h-10 text-red-600 dark:text-red-400" />
) : (
  <CheckCircle className="w-10 h-10 text-green-600 dark:text-green-400" />
)}

<motion.h2
  className={`text-5xl font-bold mb-2 ${
    isCancer ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'
  }`}
>
  {result.prediction}
</motion.h2>

<motion.p className="text-gray-700 dark:text-gray-300">
  {isCancer ? 'Suspicious findings detected' : 'No concerning findings detected'}
</motion.p>
```

---

## 4. Explainability Section Details

### Toggle State:
```javascript
// ResultsSection.jsx Line 8
const [showHeatmap, setShowHeatmap] = useState(false);

// Line 21
const hasGradcam = result?.gradcamImage;
```

### Toggle Button:
```javascript
// Lines 278-295
<motion.button
  whileHover={{ scale: 1.05 }}
  whileTap={{ scale: 0.95 }}
  onClick={() => setShowHeatmap(!showHeatmap)}
  className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-[#0077b6] to-[#00b4d8] text-white rounded-xl"
>
  {showHeatmap ? (
    <>
      <EyeOff className="w-5 h-5" />
      <span>Original</span>
    </>
  ) : (
    <>
      <Eye className="w-5 h-5" />
      <span>Heatmap</span>
    </>
  )}
</motion.button>
```

### Animated Transition:
```javascript
// Lines 298-316
<AnimatePresence mode="wait">
  <motion.div
    key={showHeatmap ? 'heatmap' : 'original'}
    initial={{ opacity: 0, scale: 0.95 }}
    animate={{ opacity: 1, scale: 1 }}
    exit={{ opacity: 0, scale: 0.95 }}
    transition={{ duration: 0.3 }}
    className="relative rounded-2xl overflow-hidden shadow-2xl"
  >
    <img
      src={showHeatmap ? result.gradcamImage : result.originalImage || result.gradcamImage}
      alt={showHeatmap ? 'Grad-CAM Heatmap' : 'Original X-ray'}
      className="w-full h-auto"
    />
    <div className="absolute top-4 left-4 px-4 py-2 bg-black/70 backdrop-blur-sm rounded-xl text-white">
      {showHeatmap ? '🔥 Heatmap View' : '📷 Original View'}
    </div>
  </motion.div>
</AnimatePresence>
```

### Info Box:
```javascript
// Lines 318-324
<div className="mt-6 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-4">
  <p className="text-sm text-gray-700 dark:text-gray-300">
    <strong>ℹ️ About Grad-CAM:</strong>{' '}
    The heatmap highlights regions that influenced the AI's decision. 
    Red/warm colors indicate areas of high importance, 
    while blue/cool colors show less significant regions.
  </p>
</div>
```

---

## 📊 Quick Reference Table:

| Feature | File | Lines | Key Code |
|---------|------|-------|----------|
| Dark Mode State | `App.jsx` | 13, 16-34 | `useState`, `useEffect`, `toggleDarkMode` |
| Dark Mode Button | `Navbar.jsx` | 96-108 | `onClick={toggleDarkMode}` |
| Grad-CAM Backend | `server.js` | 61, 108-111, 172-183, 194 | `--visualize`, `gradcamPath`, `gradcamBase64` |
| Grad-CAM Frontend | `ResultsSection.jsx` | 262-326 | `hasGradcam`, toggle, `AnimatePresence` |
| Green Glow | `ResultsSection.jsx` | 120-121, 132 | `from-green-50`, `bg-green-500` |
| Red Glow | `ResultsSection.jsx` | 120-121, 132 | `from-red-50`, `bg-red-500` |
| Pulse Animation | `ResultsSection.jsx` | 125-134 | `scale: [1, 1.2, 1]`, `repeat: Infinity` |
| Toggle Button | `ResultsSection.jsx` | 278-295 | `Eye`/`EyeOff` icons |
| Fade Transition | `ResultsSection.jsx` | 298-316 | `AnimatePresence`, `duration: 0.3` |

---

## 🔍 How to Find Code:

### Using VS Code:
1. Press `Ctrl+P` to open file search
2. Type filename (e.g., `ResultsSection.jsx`)
3. Press `Ctrl+G` to go to line number
4. Or press `Ctrl+F` to search within file

### Using Search:
1. Press `Ctrl+Shift+F` for global search
2. Search for keywords like:
   - `toggleDarkMode`
   - `gradcamImage`
   - `bg-red-500`
   - `AnimatePresence`

---

## ✅ Verification Commands:

### Check if files exist:
```bash
# Frontend components
dir frontend\src\components\*.jsx

# Backend server
dir backend\server.js

# Environment files
dir frontend\.env
dir backend\.env
```

### Check specific lines:
```bash
# PowerShell
Get-Content frontend\src\components\ResultsSection.jsx | Select-Object -First 330 -Skip 260

# Or open in editor and go to line
code frontend\src\components\ResultsSection.jsx:262
```

---

## 🎯 Summary:

**All 4 features are implemented in these files:**

1. **Dark Mode**: `App.jsx` + `Navbar.jsx`
2. **Grad-CAM**: `server.js` + `ResultsSection.jsx` + `App.jsx`
3. **Glow Effects**: `ResultsSection.jsx` (lines 118-174)
4. **Explainability**: `ResultsSection.jsx` (lines 262-326)

**Every feature has exact line numbers provided above!**
