import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import CountUp from 'react-countup';
import { AlertCircle, CheckCircle, Activity, TrendingUp, Shield, AlertTriangle, Eye, EyeOff } from 'lucide-react';

const ResultsSection = ({ result, loading }) => {
  const [showResult, setShowResult] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(false);

  useEffect(() => {
    if (result && !loading) {
      setShowResult(true);
    } else {
      setShowResult(false);
    }
  }, [result, loading]);

  const isCancer = result?.prediction === 'Cancer';
  const probability = result?.probability || 0;
  const confidence = result?.confidence || 0;
  const hasGradcam = result?.gradcamImage;

  return (
    <section className="py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <AnimatePresence mode="wait">
          {!result && !loading && (
            <motion.div
              key="empty"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl rounded-3xl shadow-2xl border border-gray-200/50 dark:border-gray-700/50 p-12 text-center"
            >
              <motion.div
                animate={{
                  scale: [1, 1.05, 1],
                  rotate: [0, 5, -5, 0],
                }}
                transition={{
                  duration: 3,
                  repeat: Infinity,
                  ease: 'easeInOut',
                }}
                className="w-32 h-32 mx-auto mb-6 bg-gradient-to-br from-[#0077b6]/10 to-[#00b4d8]/10 rounded-full flex items-center justify-center"
              >
                <Activity className="w-16 h-16 text-[#0077b6]" />
              </motion.div>
              <h3 className="text-2xl font-bold text-gray-800 dark:text-white mb-3">
                Awaiting Analysis
              </h3>
              <p className="text-gray-600 dark:text-gray-400">
                Upload an X-ray image to see AI-powered results here
              </p>
            </motion.div>
          )}

          {loading && (
            <motion.div
              key="loading"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl rounded-3xl shadow-2xl border border-gray-200/50 dark:border-gray-700/50 p-12 text-center"
            >
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                className="w-32 h-32 mx-auto mb-6 bg-gradient-to-br from-[#0077b6] to-[#00b4d8] rounded-full flex items-center justify-center relative"
              >
                <motion.div
                  animate={{
                    scale: [1, 1.2, 1],
                    opacity: [0.5, 0.8, 0.5],
                  }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                  className="absolute inset-0 bg-[#00b4d8] rounded-full blur-xl"
                />
                <Activity className="w-16 h-16 text-white relative z-10" />
              </motion.div>
              <h3 className="text-2xl font-bold text-gray-800 dark:text-white mb-3">
                Analyzing X-ray...
              </h3>
              <p className="text-gray-600 dark:text-gray-400 mb-6">
                Our AI is carefully examining the image
              </p>
              <div className="max-w-xs mx-auto">
                <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-2">
                  <span>Processing</span>
                  <span>Please wait...</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 overflow-hidden">
                  <motion.div
                    initial={{ width: '0%' }}
                    animate={{ width: '100%' }}
                    transition={{ duration: 2, ease: 'easeInOut', repeat: Infinity }}
                    className="h-full bg-gradient-to-r from-[#0077b6] to-[#00b4d8]"
                  />
                </div>
              </div>
            </motion.div>
          )}

          {showResult && result && (
            <motion.div
              key="result"
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -50 }}
              transition={{ duration: 0.6, ease: 'easeOut' }}
              className="space-y-6"
            >
              {/* Main Prediction Card */}
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
                className={`relative overflow-hidden rounded-3xl shadow-2xl p-8 ${
                  isCancer
                    ? 'bg-gradient-to-br from-red-50 to-pink-50 dark:from-red-900/20 dark:to-pink-900/20 border-2 border-red-200 dark:border-red-800'
                    : 'bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 border-2 border-green-200 dark:border-green-800'
                }`}
              >
                {/* Animated background effect */}
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

                <div className="relative z-10">
                  <div className="flex items-center justify-between mb-6">
                    <span className="text-sm font-semibold text-gray-600 dark:text-gray-400 uppercase tracking-wider">
                      Diagnosis Result
                    </span>
                    <motion.div
                      initial={{ scale: 0, rotate: -180 }}
                      animate={{ scale: 1, rotate: 0 }}
                      transition={{ delay: 0.4, type: 'spring', stiffness: 200 }}
                    >
                      {isCancer ? (
                        <AlertCircle className="w-10 h-10 text-red-600 dark:text-red-400" />
                      ) : (
                        <CheckCircle className="w-10 h-10 text-green-600 dark:text-green-400" />
                      )}
                    </motion.div>
                  </div>

                  <motion.h2
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.5 }}
                    className={`text-5xl font-bold mb-2 ${
                      isCancer ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'
                    }`}
                  >
                    {result.prediction}
                  </motion.h2>

                  <motion.p
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.6 }}
                    className="text-gray-700 dark:text-gray-300"
                  >
                    {isCancer ? 'Suspicious findings detected' : 'No concerning findings detected'}
                  </motion.p>
                </div>
              </motion.div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Probability Card */}
                <motion.div
                  initial={{ opacity: 0, x: -50 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.7 }}
                  whileHover={{ scale: 1.02, y: -5 }}
                  className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl rounded-2xl shadow-xl border border-gray-200/50 dark:border-gray-700/50 p-6"
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <div className="p-2 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl">
                        <TrendingUp className="w-5 h-5 text-white" />
                      </div>
                      <span className="font-semibold text-gray-700 dark:text-gray-300">
                        Probability
                      </span>
                    </div>
                    <motion.span
                      initial={{ opacity: 0, scale: 0 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: 1, type: 'spring', stiffness: 200 }}
                      className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent"
                    >
                      <CountUp
                        end={probability * 100}
                        duration={2}
                        decimals={1}
                        suffix="%"
                      />
                    </motion.span>
                  </div>
                  <div className="relative w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4 overflow-hidden shadow-inner">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${probability * 100}%` }}
                      transition={{ delay: 1, duration: 1.5, ease: 'easeOut' }}
                      className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full shadow-lg"
                    />
                  </div>
                </motion.div>

                {/* Confidence Card */}
                <motion.div
                  initial={{ opacity: 0, x: 50 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.8 }}
                  whileHover={{ scale: 1.02, y: -5 }}
                  className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl rounded-2xl shadow-xl border border-gray-200/50 dark:border-gray-700/50 p-6"
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      <div className="p-2 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl">
                        <Shield className="w-5 h-5 text-white" />
                      </div>
                      <span className="font-semibold text-gray-700 dark:text-gray-300">
                        Confidence
                      </span>
                    </div>
                    <motion.span
                      initial={{ opacity: 0, scale: 0 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: 1.2, type: 'spring', stiffness: 200 }}
                      className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent"
                    >
                      <CountUp
                        end={confidence * 100}
                        duration={2}
                        decimals={1}
                        suffix="%"
                      />
                    </motion.span>
                  </div>
                  <div className="relative w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4 overflow-hidden shadow-inner">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${confidence * 100}%` }}
                      transition={{ delay: 1.2, duration: 1.5, ease: 'easeOut' }}
                      className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full shadow-lg"
                    />
                  </div>
                </motion.div>
              </div>

              {/* Grad-CAM Visualization Section */}
              {hasGradcam && (
                <motion.div
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.9 }}
                  className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl rounded-3xl shadow-2xl border border-gray-200/50 dark:border-gray-700/50 p-8"
                >
                  <div className="flex items-center justify-between mb-6">
                    <div>
                      <h3 className="text-2xl font-bold text-gray-800 dark:text-white mb-2">
                        AI Explainability
                      </h3>
                      <p className="text-gray-600 dark:text-gray-400">
                        Grad-CAM heatmap showing regions the AI focused on
                      </p>
                    </div>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => setShowHeatmap(!showHeatmap)}
                      className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-[#0077b6] to-[#00b4d8] text-white rounded-xl font-semibold shadow-lg"
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
                  </div>

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
                      <div className="absolute top-4 left-4 px-4 py-2 bg-black/70 backdrop-blur-sm rounded-xl text-white font-semibold">
                        {showHeatmap ? '🔥 Heatmap View' : '📷 Original View'}
                      </div>
                    </motion.div>
                  </AnimatePresence>

                  <div className="mt-6 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-4">
                    <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
                      <strong className="text-blue-900 dark:text-blue-300">ℹ️ About Grad-CAM:</strong>{' '}
                      The heatmap highlights regions that influenced the AI's decision. Red/warm colors indicate areas of high importance, 
                      while blue/cool colors show less significant regions.
                    </p>
                  </div>
                </motion.div>
              )}

              {/* Medical Recommendation */}
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: hasGradcam ? 1.1 : 0.9 }}
                className="bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 border-2 border-amber-200 dark:border-amber-800 rounded-2xl p-6 shadow-xl"
              >
                <div className="flex items-start space-x-4">
                  <motion.div
                    animate={{ rotate: [0, 10, -10, 0] }}
                    transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
                    className="flex-shrink-0 p-3 bg-gradient-to-br from-amber-500 to-orange-500 rounded-xl"
                  >
                    <AlertTriangle className="w-6 h-6 text-white" />
                  </motion.div>
                  <div>
                    <h4 className="text-lg font-bold text-amber-900 dark:text-amber-300 mb-2">
                      Medical Recommendation
                    </h4>
                    <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
                      {result.message}
                    </p>
                  </div>
                </div>
              </motion.div>

              {/* Disclaimer */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1 }}
                className="bg-gray-50 dark:bg-gray-800/50 rounded-2xl p-6 border border-gray-200 dark:border-gray-700"
              >
                <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                  <strong className="text-gray-800 dark:text-gray-200">⚠️ Important Disclaimer:</strong>{' '}
                  This AI system is designed to assist medical professionals and should not be used as the sole basis for diagnosis. 
                  Always consult with qualified healthcare providers for medical advice, diagnosis, and treatment decisions.
                </p>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
};

export default ResultsSection;
