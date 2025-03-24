'use client';
import React from 'react';
import { motion } from 'framer-motion';
import { fadeIn } from './animations';

interface CacheStatsProps {
  cacheStats: {
    hits?: number;
    misses?: number;
    money_saved?: number;
    [key: string]: any;
  };
}

export default function CacheStats({ cacheStats }: CacheStatsProps) {
  if (!cacheStats) {
    return null;
  }

  const totalRequests = (cacheStats.hits || 0) + (cacheStats.misses || 0);
  const hitRate = totalRequests > 0 ? (cacheStats.hits || 0) / totalRequests * 100 : 0;
  
  return (
    <motion.div 
      className="p-4 border rounded-lg bg-muted/20 shadow-sm"
      initial="hidden"
      animate="visible"
      variants={fadeIn}
    >
      <motion.h3 
        className="text-lg font-medium mb-3"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        Cache Performance
      </motion.h3>
      
      <motion.div 
        className="grid grid-cols-2 md:grid-cols-4 gap-3"
      >
        <motion.div 
          className="bg-muted/30 p-3 rounded-md text-center"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3, delay: 0.1 }}
          whileHover={{ y: -2, scale: 1.03 }}
        >
          <motion.div 
            className="text-2xl font-bold"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            {cacheStats.hits || 0}
          </motion.div>
          <div className="text-xs text-muted-foreground">Cache Hits</div>
        </motion.div>
        
        <motion.div 
          className="bg-muted/30 p-3 rounded-md text-center"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3, delay: 0.2 }}
          whileHover={{ y: -2, scale: 1.03 }}
        >
          <motion.div 
            className="text-2xl font-bold"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            {cacheStats.misses || 0}
          </motion.div>
          <div className="text-xs text-muted-foreground">Cache Misses</div>
        </motion.div>
        
        <motion.div 
          className="bg-muted/30 p-3 rounded-md text-center"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3, delay: 0.3 }}
          whileHover={{ y: -2, scale: 1.03 }}
        >
          <motion.div 
            className="text-2xl font-bold"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            {hitRate.toFixed(1)}%
          </motion.div>
          <div className="text-xs text-muted-foreground">Hit Rate</div>
        </motion.div>
        
        <motion.div 
          className="bg-green-100 dark:bg-green-900/30 p-3 rounded-md text-center"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3, delay: 0.4 }}
          whileHover={{ y: -2, scale: 1.03 }}
        >
          <motion.div 
            className="text-2xl font-bold text-green-700 dark:text-green-400"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.5 }}
          >
            ${(cacheStats.money_saved || 0).toFixed(2)}
          </motion.div>
          <div className="text-xs text-green-600 dark:text-green-500">Cost Savings</div>
        </motion.div>
      </motion.div>
      
      {/* Cache efficiency visualization */}
      {totalRequests > 0 && (
        <motion.div 
          className="mt-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.6 }}
        >
          <div className="text-xs text-muted-foreground mb-1">Cache Efficiency</div>
          <div className="w-full h-3 bg-muted/50 rounded-full overflow-hidden">
            <motion.div 
              className="h-full bg-primary" 
              initial={{ width: 0 }}
              animate={{ width: `${hitRate}%` }}
              transition={{ duration: 1, delay: 0.8, ease: "easeOut" }}
            />
          </div>
          <div className="flex justify-between text-xs mt-1">
            <span>0%</span>
            <span>50%</span>
            <span>100%</span>
          </div>
        </motion.div>
      )}
      
      {/* Cost savings explanation */}
      {(cacheStats.money_saved || 0) > 0 && (
        <motion.div 
          className="mt-4 text-xs text-muted-foreground"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.9 }}
        >
          <p>
            Semantic caching saved ${(cacheStats.money_saved || 0).toFixed(2)} by reusing similar images 
            instead of generating new ones. This reduces both cost and generation time.
          </p>
        </motion.div>
      )}
    </motion.div>
  );
}