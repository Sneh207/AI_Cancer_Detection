import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import { Upload, X, AlertCircle, Loader, Image as ImageIcon, CheckCircle } from 'lucide-react';

const UploadSection = ({ onAnalyze, loading, error }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg'],
    },
    maxFiles: 1,
    onDragEnter: () => setIsDragging(true),
    onDragLeave: () => setIsDragging(false),
    onDropAccepted: () => setIsDragging(false),
  });

  const handleClear = () => {
    setSelectedFile(null);
    setPreview(null);
  };

  const handleAnalyzeClick = () => {
    if (selectedFile) {
      onAnalyze(selectedFile);
    }
  };

  return (
    <section id="upload" className="py-20 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <motion.div
            initial={{ scale: 0 }}
            whileInView={{ scale: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
            className="inline-block mb-4"
          >
            <div className="p-4 bg-gradient-to-br from-[#0077b6]/10 to-[#00b4d8]/10 rounded-2xl">
              <Upload className="w-8 h-8 text-[#0077b6]" />
            </div>
          </motion.div>
          <h2 className="text-4xl md:text-5xl font-bold text-gray-800 dark:text-white mb-4">
            Upload Your X-ray
          </h2>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
            Drag and drop your chest X-ray image or click to browse
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Card */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl rounded-3xl shadow-2xl border border-gray-200/50 dark:border-gray-700/50 p-8">
              <AnimatePresence mode="wait">
                {!preview ? (
                  <motion.div
                    key="dropzone"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    {...getRootProps()}
                    className="cursor-pointer"
                  >
                    <input {...getInputProps()} />
                    <motion.div
                      animate={{
                        borderColor: isDragging ? '#00b4d8' : '#e5e7eb',
                        backgroundColor: isDragging ? 'rgba(0, 180, 216, 0.05)' : 'transparent',
                      }}
                      className="border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300"
                    >
                      <motion.div
                        animate={{
                          scale: isDragging ? 1.1 : 1,
                          rotate: isDragging ? [0, -5, 5, 0] : 0,
                        }}
                        transition={{ duration: 0.3 }}
                        className="relative"
                      >
                        <div className="w-24 h-24 mx-auto mb-6 bg-gradient-to-br from-[#0077b6]/10 to-[#00b4d8]/10 rounded-full flex items-center justify-center">
                          <ImageIcon className="w-12 h-12 text-[#0077b6]" />
                        </div>
                        
                        {isDragging && (
                          <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            className="absolute inset-0 bg-[#00b4d8]/10 rounded-full blur-2xl"
                          />
                        )}
                      </motion.div>

                      <p className="text-lg font-semibold text-gray-700 dark:text-gray-300 mb-2">
                        {isDragging ? 'Drop your X-ray here' : 'Click to upload or drag and drop'}
                      </p>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        PNG, JPG up to 10MB
                      </p>

                      <motion.div
                        whileHover={{ scale: 1.05 }}
                        className="mt-6 inline-block px-6 py-3 bg-gradient-to-r from-[#0077b6] to-[#00b4d8] text-white rounded-xl font-semibold shadow-lg"
                      >
                        Browse Files
                      </motion.div>
                    </motion.div>
                  </motion.div>
                ) : (
                  <motion.div
                    key="preview"
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.9 }}
                    className="relative"
                  >
                    <div className="relative rounded-2xl overflow-hidden shadow-2xl">
                      <img
                        src={preview}
                        alt="X-ray preview"
                        className="w-full h-auto"
                      />
                      
                      {/* Scanning Animation Overlay */}
                      {loading && (
                        <motion.div
                          initial={{ y: '-100%' }}
                          animate={{ y: '200%' }}
                          transition={{
                            duration: 1.5,
                            repeat: Infinity,
                            ease: 'linear',
                          }}
                          className="absolute inset-0 bg-gradient-to-b from-transparent via-[#00b4d8]/40 to-transparent h-32 pointer-events-none"
                        />
                      )}
                    </div>

                    <motion.button
                      whileHover={{ scale: 1.1, rotate: 90 }}
                      whileTap={{ scale: 0.9 }}
                      onClick={handleClear}
                      className="absolute top-4 right-4 p-3 bg-red-500 hover:bg-red-600 text-white rounded-full shadow-lg transition-colors"
                    >
                      <X className="w-5 h-5" />
                    </motion.button>

                    {!loading && (
                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="mt-4 flex items-center justify-center space-x-2 text-green-600 dark:text-green-400"
                      >
                        <CheckCircle className="w-5 h-5" />
                        <span className="font-medium">Image loaded successfully</span>
                      </motion.div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Error Message */}
              <AnimatePresence>
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="mt-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-4 flex items-start"
                  >
                    <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 mr-3 flex-shrink-0 mt-0.5" />
                    <p className="text-red-700 dark:text-red-300 text-sm">{error}</p>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Analyze Button */}
              <motion.button
                whileHover={{ scale: selectedFile && !loading ? 1.02 : 1 }}
                whileTap={{ scale: selectedFile && !loading ? 0.98 : 1 }}
                onClick={handleAnalyzeClick}
                disabled={!selectedFile || loading}
                className={`mt-6 w-full py-5 px-6 rounded-2xl font-bold text-lg transition-all duration-300 flex items-center justify-center space-x-3 shadow-xl ${
                  !selectedFile || loading
                    ? 'bg-gray-200 dark:bg-gray-700 text-gray-400 dark:text-gray-500 cursor-not-allowed'
                    : 'bg-gradient-to-r from-[#0077b6] to-[#00b4d8] hover:from-[#0077b6] hover:to-[#0096c7] text-white'
                }`}
              >
                {loading ? (
                  <>
                    <Loader className="w-6 h-6 animate-spin" />
                    <span>Analyzing...</span>
                  </>
                ) : (
                  <>
                    <Upload className="w-6 h-6" />
                    <span>Analyze X-ray</span>
                  </>
                )}
              </motion.button>

              {/* Progress Bar */}
              {loading && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="mt-4"
                >
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2 overflow-hidden">
                    <motion.div
                      initial={{ width: '0%' }}
                      animate={{ width: '100%' }}
                      transition={{ duration: 2, ease: 'easeInOut' }}
                      className="h-full bg-gradient-to-r from-[#0077b6] to-[#00b4d8]"
                    />
                  </div>
                  <p className="text-center text-sm text-gray-600 dark:text-gray-400 mt-2">
                    AI is analyzing your X-ray...
                  </p>
                </motion.div>
              )}
            </div>
          </motion.div>

          {/* Info Card */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="space-y-6"
          >
            {/* How it Works */}
            <div className="bg-gradient-to-br from-[#caf0f8]/50 to-white dark:from-gray-800/50 dark:to-gray-900/50 backdrop-blur-xl rounded-3xl shadow-xl border border-[#00b4d8]/20 p-8">
              <h3 className="text-2xl font-bold text-gray-800 dark:text-white mb-6">
                How It Works
              </h3>
              <div className="space-y-4">
                {[
                  { step: 1, title: 'Upload Image', desc: 'Select a chest X-ray in JPG or PNG format', color: 'from-blue-500 to-cyan-500' },
                  { step: 2, title: 'AI Analysis', desc: 'Our deep learning model analyzes the image', color: 'from-purple-500 to-pink-500' },
                  { step: 3, title: 'Get Results', desc: 'Receive confidence scores and recommendations', color: 'from-green-500 to-emerald-500' },
                  { step: 4, title: 'Consult Doctor', desc: 'Always verify with a medical professional', color: 'from-orange-500 to-red-500' },
                ].map((item, index) => (
                  <motion.div
                    key={item.step}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: index * 0.1 }}
                    whileHover={{ x: 5 }}
                    className="flex items-start space-x-4 p-4 rounded-xl hover:bg-white/50 dark:hover:bg-gray-800/50 transition-all duration-300"
                  >
                    <div className={`flex-shrink-0 w-10 h-10 rounded-xl bg-gradient-to-br ${item.color} flex items-center justify-center text-white font-bold shadow-lg`}>
                      {item.step}
                    </div>
                    <div>
                      <h4 className="font-semibold text-gray-800 dark:text-white mb-1">
                        {item.title}
                      </h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {item.desc}
                      </p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 gap-4">
              {[
                { value: '96%', label: 'Accuracy', color: 'from-green-500 to-emerald-500' },
                { value: '<5m', label: 'Analysis Time', color: 'from-blue-500 to-cyan-500' },
              ].map((stat, index) => (
                <motion.div
                  key={stat.label}
                  initial={{ opacity: 0, scale: 0.8 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                  whileHover={{ scale: 1.05, y: -5 }}
                  className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl rounded-2xl shadow-xl border border-gray-200/50 dark:border-gray-700/50 p-6 text-center"
                >
                  <div className={`text-3xl font-bold bg-gradient-to-r ${stat.color} bg-clip-text text-transparent mb-2`}>
                    {stat.value}
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400 font-medium">
                    {stat.label}
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default UploadSection;
