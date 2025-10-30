import React from 'react';
import { motion } from 'framer-motion';
import { ArrowDown, Sparkles, Shield, Zap } from 'lucide-react';

const HeroSection = ({ onGetStarted }) => {
  const floatingAnimation = {
    y: [0, -20, 0],
    transition: {
      duration: 3,
      repeat: Infinity,
      ease: 'easeInOut',
    },
  };

  const pulseAnimation = {
    scale: [1, 1.05, 1],
    opacity: [0.5, 0.8, 0.5],
    transition: {
      duration: 2,
      repeat: Infinity,
      ease: 'easeInOut',
    },
  };

  return (
    <section id="hero" className="relative min-h-screen flex items-center justify-center overflow-hidden pt-20">
      {/* Animated Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-[#caf0f8] via-white to-[#00b4d8]/20 dark:from-gray-900 dark:via-gray-800 dark:to-[#0077b6]/20">
        {/* Floating Particles */}
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-2 h-2 bg-[#00b4d8]/30 rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
            }}
            animate={{
              y: [0, -30, 0],
              x: [0, Math.random() * 20 - 10, 0],
              opacity: [0.2, 0.5, 0.2],
            }}
            transition={{
              duration: 3 + Math.random() * 2,
              repeat: Infinity,
              delay: Math.random() * 2,
            }}
          />
        ))}

        {/* Gradient Orbs */}
        <motion.div
          animate={pulseAnimation}
          className="absolute top-20 left-10 w-96 h-96 bg-gradient-to-br from-[#0077b6]/20 to-[#00b4d8]/20 rounded-full blur-3xl"
        />
        <motion.div
          animate={{
            ...pulseAnimation,
            transition: { ...pulseAnimation.transition, delay: 1 },
          }}
          className="absolute bottom-20 right-10 w-96 h-96 bg-gradient-to-br from-[#00b4d8]/20 to-[#caf0f8]/30 rounded-full blur-3xl"
        />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="space-y-8"
        >
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
            className="inline-flex items-center space-x-2 bg-white/80 dark:bg-gray-800/80 backdrop-blur-lg px-6 py-3 rounded-full shadow-lg border border-[#00b4d8]/20"
          >
            <Sparkles className="w-4 h-4 text-[#0077b6]" />
            <span className="text-sm font-semibold text-gray-700 dark:text-gray-300">
              AI-Powered Medical Analysis
            </span>
          </motion.div>

          {/* Main Title */}
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.8 }}
            className="text-5xl md:text-7xl font-bold leading-tight"
          >
            <span className="bg-gradient-to-r from-[#0077b6] via-[#00b4d8] to-[#0077b6] bg-clip-text text-transparent">
              Early Detection
            </span>
            <br />
            <span className="text-gray-800 dark:text-white">Saves Lives</span>
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.8 }}
            className="text-xl md:text-2xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto leading-relaxed"
          >
            Upload your chest X-ray and let our advanced AI analyze it for signs of cancer.
            <br />
            <span className="text-[#0077b6] font-semibold">Fast. Accurate. Reliable.</span>
          </motion.p>

          {/* Feature Pills */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="flex flex-wrap justify-center gap-4 pt-4"
          >
            {[
              { icon: Shield, text: 'HIPAA Compliant', color: 'from-green-500 to-emerald-500' },
              { icon: Zap, text: 'Instant Results', color: 'from-yellow-500 to-orange-500' },
              { icon: Sparkles, text: '95% Accuracy', color: 'from-purple-500 to-pink-500' },
            ].map((feature, index) => (
              <motion.div
                key={feature.text}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.6 + index * 0.1 }}
                whileHover={{ scale: 1.05, y: -2 }}
                className="flex items-center space-x-2 bg-white/90 dark:bg-gray-800/90 backdrop-blur-lg px-5 py-3 rounded-full shadow-lg border border-gray-200/50 dark:border-gray-700/50"
              >
                <div className={`p-1.5 rounded-lg bg-gradient-to-br ${feature.color}`}>
                  <feature.icon className="w-4 h-4 text-white" />
                </div>
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {feature.text}
                </span>
              </motion.div>
            ))}
          </motion.div>

          {/* CTA Button */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
            className="pt-8"
          >
            <motion.button
              whileHover={{ scale: 1.05, boxShadow: '0 20px 40px rgba(0, 119, 182, 0.3)' }}
              whileTap={{ scale: 0.95 }}
              onClick={onGetStarted}
              className="group relative px-10 py-5 bg-gradient-to-r from-[#0077b6] to-[#00b4d8] text-white rounded-2xl font-bold text-lg shadow-2xl overflow-hidden"
            >
              <motion.div
                className="absolute inset-0 bg-gradient-to-r from-[#00b4d8] to-[#0077b6]"
                initial={{ x: '100%' }}
                whileHover={{ x: 0 }}
                transition={{ duration: 0.3 }}
              />
              <span className="relative z-10 flex items-center space-x-2">
                <span>Get Started</span>
                <motion.div
                  animate={{ x: [0, 5, 0] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  <ArrowDown className="w-5 h-5" />
                </motion.div>
              </span>
            </motion.button>
          </motion.div>

          {/* Floating X-ray Illustration */}
          <motion.div
            animate={floatingAnimation}
            className="pt-16"
          >
            <div className="relative w-64 h-64 mx-auto">
              {/* Glowing effect */}
              <motion.div
                animate={pulseAnimation}
                className="absolute inset-0 bg-gradient-to-br from-[#0077b6]/30 to-[#00b4d8]/30 rounded-3xl blur-2xl"
              />
              
              {/* X-ray placeholder */}
              <div className="relative bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-3xl shadow-2xl border border-gray-200/50 dark:border-gray-700/50 p-8 flex items-center justify-center">
                <svg
                  className="w-full h-full text-[#0077b6]/50"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                  />
                </svg>
              </div>

              {/* Scanning effect */}
              <motion.div
                animate={{
                  y: ['-100%', '200%'],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  ease: 'linear',
                }}
                className="absolute inset-0 bg-gradient-to-b from-transparent via-[#00b4d8]/30 to-transparent h-20 pointer-events-none"
              />
            </div>
          </motion.div>
        </motion.div>
      </div>

      {/* Scroll Indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1 }}
        className="absolute bottom-10 left-1/2 transform -translate-x-1/2"
      >
        <motion.div
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 1.5, repeat: Infinity }}
          className="text-gray-400 dark:text-gray-600"
        >
          <ArrowDown className="w-6 h-6" />
        </motion.div>
      </motion.div>
    </section>
  );
};

export default HeroSection;
