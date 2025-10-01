import React, { useState } from 'react';
import { Upload, X, AlertCircle, CheckCircle, Loader, Heart } from 'lucide-react';

const CancerDetectionApp = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setError(null);
      setResult(null);
      
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    } else {
      setError('Please select a valid image file (JPG, PNG)');
    }
  };

  const handleClearImage = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setResult(data.result);
      } else {
        setError(data.error || 'Analysis failed');
      }
    } catch (err) {
      setError('Failed to connect to server. Please ensure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-purple-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-lg border-b border-gray-200 sticky top-0 z-10 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center shadow-lg">
                <Heart className="w-6 h-6 text-white" fill="white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Cancer Detection AI
                </h1>
                <p className="text-sm text-gray-600">Chest X-ray Analysis System</p>
              </div>
            </div>
            <div className="flex items-center space-x-2 text-sm">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-gray-600 font-medium">System Active</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="space-y-6">
            <div className="bg-white rounded-3xl shadow-xl border border-gray-200 p-8">
              <h2 className="text-xl font-semibold mb-6 flex items-center text-gray-800">
                <Upload className="w-5 h-5 mr-2 text-blue-600" />
                Upload X-ray Image
              </h2>

              {!preview ? (
                <label className="cursor-pointer block">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                  <div className="border-2 border-dashed border-gray-300 rounded-2xl p-12 text-center hover:border-blue-400 hover:bg-blue-50/50 transition-all duration-300">
                    <div className="w-20 h-20 bg-gradient-to-br from-blue-100 to-purple-100 rounded-full mx-auto mb-4 flex items-center justify-center">
                      <Upload className="w-10 h-10 text-blue-600" />
                    </div>
                    <p className="text-gray-700 font-medium mb-2">
                      Click to upload or drag and drop
                    </p>
                    <p className="text-sm text-gray-500">
                      PNG, JPG up to 10MB
                    </p>
                  </div>
                </label>
              ) : (
                <div className="relative">
                  <img
                    src={preview}
                    alt="Preview"
                    className="w-full rounded-2xl border-2 border-gray-200 shadow-lg"
                  />
                  <button
                    onClick={handleClearImage}
                    className="absolute top-3 right-3 bg-red-500 hover:bg-red-600 text-white p-2 rounded-xl transition-colors shadow-lg"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>
              )}

              {error && (
                <div className="mt-4 bg-red-50 border border-red-200 rounded-xl p-4 flex items-start">
                  <AlertCircle className="w-5 h-5 text-red-600 mr-3 flex-shrink-0 mt-0.5" />
                  <p className="text-red-700 text-sm">{error}</p>
                </div>
              )}

              <button
                onClick={handleAnalyze}
                disabled={!selectedFile || loading}
                className={`mt-6 w-full py-4 px-6 rounded-xl font-semibold transition-all duration-300 flex items-center justify-center space-x-2 shadow-lg ${
                  !selectedFile || loading
                    ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                    : 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white transform hover:scale-105'
                }`}
              >
                {loading ? (
                  <>
                    <Loader className="w-5 h-5 animate-spin" />
                    <span>Analyzing...</span>
                  </>
                ) : (
                  <span>Analyze X-ray</span>
                )}
              </button>
            </div>

            {/* Information Card */}
            <div className="bg-gradient-to-br from-blue-50 to-purple-50 border border-blue-200 rounded-3xl p-6 shadow-lg">
              <h3 className="font-semibold mb-3 text-blue-900">How it works</h3>
              <ul className="space-y-2 text-sm text-gray-700">
                <li className="flex items-start">
                  <span className="text-blue-600 mr-2 font-bold">1.</span>
                  Upload a chest X-ray image in JPG or PNG format
                </li>
                <li className="flex items-start">
                  <span className="text-purple-600 mr-2 font-bold">2.</span>
                  Our AI analyzes the image for signs of cancer
                </li>
                <li className="flex items-start">
                  <span className="text-pink-600 mr-2 font-bold">3.</span>
                  Results are provided with confidence scores
                </li>
                <li className="flex items-start">
                  <span className="text-indigo-600 mr-2 font-bold">4.</span>
                  Always consult a medical professional
                </li>
              </ul>
            </div>
          </div>

          {/* Results Section */}
          <div>
            <div className="bg-white rounded-3xl shadow-xl border border-gray-200 p-8 sticky top-24">
              <h2 className="text-xl font-semibold mb-6 text-gray-800">Analysis Results</h2>

              {!result && !loading && (
                <div className="text-center py-16">
                  <div className="w-24 h-24 bg-gradient-to-br from-blue-100 to-purple-100 rounded-full mx-auto mb-4 flex items-center justify-center">
                    <svg className="w-12 h-12 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <p className="text-gray-600 font-medium">
                    Upload and analyze an X-ray to see results here
                  </p>
                </div>
              )}

              {loading && (
                <div className="text-center py-16">
                  <div className="w-24 h-24 bg-gradient-to-br from-blue-100 to-purple-100 rounded-full mx-auto mb-4 flex items-center justify-center">
                    <Loader className="w-12 h-12 text-blue-600 animate-spin" />
                  </div>
                  <p className="text-gray-700 font-medium">Analyzing image...</p>
                  <p className="text-sm text-gray-500 mt-2">This may take a few seconds</p>
                </div>
              )}

              {result && (
                <div className="space-y-6">
                  {/* Prediction Badge */}
                  <div className={`rounded-2xl p-6 shadow-lg ${
                    result.prediction === 'Cancer'
                      ? 'bg-gradient-to-br from-red-50 to-pink-50 border-2 border-red-200'
                      : 'bg-gradient-to-br from-green-50 to-emerald-50 border-2 border-green-200'
                  }`}>
                    <div className="flex items-center justify-between mb-4">
                      <span className="text-sm text-gray-600 font-medium">Prediction</span>
                      {result.prediction === 'Cancer' ? (
                        <AlertCircle className="w-6 h-6 text-red-600" />
                      ) : (
                        <CheckCircle className="w-6 h-6 text-green-600" />
                      )}
                    </div>
                    <h3 className={`text-3xl font-bold ${
                      result.prediction === 'Cancer' ? 'text-red-600' : 'text-green-600'
                    }`}>
                      {result.prediction}
                    </h3>
                  </div>

                  {/* Metrics */}
                  <div className="space-y-4">
                    <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-2xl p-5 shadow">
                      <div className="flex justify-between items-center mb-3">
                        <span className="text-sm text-gray-700 font-medium">Probability</span>
                        <span className="text-xl font-bold text-blue-700">
                          {(result.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-white rounded-full h-3 shadow-inner">
                        <div
                          className="bg-gradient-to-r from-blue-500 to-blue-600 h-3 rounded-full transition-all duration-500 shadow"
                          style={{ width: `${result.probability * 100}%` }}
                        ></div>
                      </div>
                    </div>

                    <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-2xl p-5 shadow">
                      <div className="flex justify-between items-center mb-3">
                        <span className="text-sm text-gray-700 font-medium">Confidence</span>
                        <span className="text-xl font-bold text-purple-700">
                          {(result.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-white rounded-full h-3 shadow-inner">
                        <div
                          className="bg-gradient-to-r from-purple-500 to-purple-600 h-3 rounded-full transition-all duration-500 shadow"
                          style={{ width: `${result.confidence * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>

                  {/* Medical Advice */}
                  <div className="bg-gradient-to-br from-amber-50 to-orange-50 border-2 border-amber-200 rounded-2xl p-5 shadow-lg">
                    <p className="text-sm text-amber-900 font-semibold mb-2">
                      Medical Recommendation
                    </p>
                    <p className="text-sm text-gray-700 leading-relaxed">
                      {result.message}
                    </p>
                  </div>

                  {/* Disclaimer */}
                  <div className="bg-gray-50 rounded-2xl p-5 border border-gray-200">
                    <p className="text-xs text-gray-600 leading-relaxed">
                      <strong className="text-gray-800">Disclaimer:</strong> This AI system is designed to assist medical professionals and should not be used as the sole basis for diagnosis. Always consult with qualified healthcare providers for medical advice and treatment decisions.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-12 py-6 bg-white/50 backdrop-blur border-t border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-sm text-gray-600">
          <p className="font-medium">Cancer Detection AI System</p>
          <p className="mt-1">Developed by Sneh Gupta and Arpit Bhardwaj • CSET211 - Statistical Machine Learning</p>
        </div>
      </footer>
    </div>
  );
};

export default CancerDetectionApp;