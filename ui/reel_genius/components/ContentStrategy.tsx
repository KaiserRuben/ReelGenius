"use client"
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { fadeIn, slideUp } from './animations';

interface ContentStrategyProps {
  contentAnalysis?: any;
  contentStrategy?: any;
  visualPlan?: any;
}

export default function ContentStrategy({ contentAnalysis, contentStrategy, visualPlan }: ContentStrategyProps) {
  const [activeTab, setActiveTab] = useState<string>('strategy');
  
  if (!contentStrategy && !contentAnalysis && !visualPlan) {
    return (
      <div className="p-4 text-center text-muted-foreground">
        No content strategy data available
      </div>
    );
  }

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={fadeIn}
    >
      {/* Tab navigation */}
      <motion.div 
        className="border-b flex space-x-1 mb-4"
        variants={slideUp}
      >
        {contentStrategy && (
          <motion.button
            className={`px-4 py-2 text-sm font-medium relative ${
              activeTab === 'strategy'
                ? 'text-primary'
                : 'text-muted-foreground hover:text-foreground'
            }`}
            onClick={() => setActiveTab('strategy')}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Strategy
            {activeTab === 'strategy' && (
              <motion.div 
                className="absolute bottom-0 left-0 w-full h-0.5 bg-primary"
                layoutId="activeTab"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3 }}
              />
            )}
          </motion.button>
        )}
        
        {contentAnalysis && (
          <motion.button
            className={`px-4 py-2 text-sm font-medium relative ${
              activeTab === 'analysis'
                ? 'text-primary'
                : 'text-muted-foreground hover:text-foreground'
            }`}
            onClick={() => setActiveTab('analysis')}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Analysis
            {activeTab === 'analysis' && (
              <motion.div 
                className="absolute bottom-0 left-0 w-full h-0.5 bg-primary"
                layoutId="activeTab"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3 }}
              />
            )}
          </motion.button>
        )}
        
        {visualPlan && (
          <motion.button
            className={`px-4 py-2 text-sm font-medium relative ${
              activeTab === 'visual'
                ? 'text-primary'
                : 'text-muted-foreground hover:text-foreground'
            }`}
            onClick={() => setActiveTab('visual')}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Visual Plan
            {activeTab === 'visual' && (
              <motion.div 
                className="absolute bottom-0 left-0 w-full h-0.5 bg-primary"
                layoutId="activeTab"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3 }}
              />
            )}
          </motion.button>
        )}
      </motion.div>
      
      {/* Content Sections */}
      <AnimatePresence mode="wait">
        {/* Content Strategy */}
        {activeTab === 'strategy' && contentStrategy && (
          <motion.div 
            className="space-y-6"
            key="strategy"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
          >
            {/* Hook engineering */}
          {contentStrategy.hook_engineering && (
            <div className="p-4 bg-muted/20 rounded-md">
              <h3 className="text-lg font-medium mb-2">Hook Engineering</h3>
              <p className="text-sm mb-3">{contentStrategy.hook_engineering.approach}</p>
              
              {contentStrategy.hook_engineering.elements && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-2">
                  {contentStrategy.hook_engineering.elements.map((element: any, index: number) => (
                    <div key={index} className="p-2 bg-muted/30 rounded">
                      <p className="text-xs">{element}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
          
          {/* Narrative architecture */}
          {contentStrategy.narrative_architecture && (
            <div className="p-4 bg-muted/20 rounded-md">
              <h3 className="text-lg font-medium mb-2">Narrative Architecture</h3>
              <p className="text-sm">{contentStrategy.narrative_architecture.structure}</p>
              
              {contentStrategy.narrative_architecture.pacing && (
                <div className="mt-2">
                  <h4 className="text-sm font-medium">Pacing</h4>
                  <p className="text-xs">{contentStrategy.narrative_architecture.pacing}</p>
                </div>
              )}
            </div>
          )}
          
          {/* Algorithm Optimization */}
          {contentStrategy.algorithm_optimization && (
            <div className="p-4 bg-muted/20 rounded-md">
              <h3 className="text-lg font-medium mb-2">Algorithm Optimization</h3>
              <p className="text-sm mb-3">{contentStrategy.algorithm_optimization.approach}</p>
              
              {contentStrategy.algorithm_optimization.metrics && (
                <div className="mt-2">
                  <h4 className="text-sm font-medium">Key Metrics</h4>
                  <ul className="list-disc list-inside space-y-1 mt-1">
                    {contentStrategy.algorithm_optimization.metrics.map((metric: string, index: number) => (
                      <li key={index} className="text-xs">{metric}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
          
          {/* Engagement Engineering */}
          {contentStrategy.engagement_engineering && (
            <div className="p-4 bg-muted/20 rounded-md">
              <h3 className="text-lg font-medium mb-2">Engagement Engineering</h3>
              <p className="text-sm">{contentStrategy.engagement_engineering.approach}</p>
              
              {contentStrategy.engagement_engineering.techniques && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mt-3">
                  {contentStrategy.engagement_engineering.techniques.map((tech: string, index: number) => (
                    <div key={index} className="p-2 bg-muted/30 rounded">
                      <p className="text-xs">{tech}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </motion.div>
      )}
      
      {/* Content Analysis */}
      {activeTab === 'analysis' && contentAnalysis && (
        <motion.div 
          className="space-y-6"
          key="analysis"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.3 }}
        >
          {/* Topic Analysis */}
          <div className="p-4 bg-muted/20 rounded-md">
            <h3 className="text-lg font-medium mb-2">Topic Analysis</h3>
            <p className="text-sm">{contentAnalysis.topic_summary || contentAnalysis.topic || 'No topic summary available'}</p>
            
            {contentAnalysis.keywords && (
              <div className="mt-3">
                <h4 className="text-sm font-medium mb-1">Keywords</h4>
                <div className="flex flex-wrap gap-1">
                  {contentAnalysis.keywords.map((keyword: string, index: number) => (
                    <span 
                      key={index} 
                      className="px-2 py-1 text-xs bg-muted/40 rounded-full"
                    >
                      {keyword}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
          
          {/* Audience Analysis */}
          {contentAnalysis.audience_analysis && (
            <div className="p-4 bg-muted/20 rounded-md">
              <h3 className="text-lg font-medium mb-2">Audience Analysis</h3>
              <p className="text-sm">{contentAnalysis.audience_analysis.description}</p>
              
              {contentAnalysis.audience_analysis.demographics && (
                <div className="mt-3">
                  <h4 className="text-sm font-medium mb-1">Demographics</h4>
                  <p className="text-xs">{contentAnalysis.audience_analysis.demographics}</p>
                </div>
              )}
              
              {contentAnalysis.audience_analysis.interests && (
                <div className="mt-3">
                  <h4 className="text-sm font-medium mb-1">Interests</h4>
                  <div className="flex flex-wrap gap-1">
                    {contentAnalysis.audience_analysis.interests.map((interest: string, index: number) => (
                      <span 
                        key={index} 
                        className="px-2 py-1 text-xs bg-muted/40 rounded-full"
                      >
                        {interest}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
          
          {/* Trend Analysis */}
          {contentAnalysis.trend_analysis && (
            <div className="p-4 bg-muted/20 rounded-md">
              <h3 className="text-lg font-medium mb-2">Trend Analysis</h3>
              <p className="text-sm">{contentAnalysis.trend_analysis.overview}</p>
              
              {contentAnalysis.trend_analysis.current_trends && (
                <div className="mt-3">
                  <h4 className="text-sm font-medium mb-1">Current Trends</h4>
                  <ul className="list-disc list-inside space-y-1">
                    {contentAnalysis.trend_analysis.current_trends.map((trend: string, index: number) => (
                      <li key={index} className="text-xs">{trend}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </motion.div>
      )}
      
      {/* Visual Plan */}
      {activeTab === 'visual' && visualPlan && (
        <motion.div 
          className="space-y-6"
          key="visual"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.3 }}
        >
          {/* Visual Style */}
          {visualPlan.style && (
            <div className="p-4 bg-muted/20 rounded-md">
              <h3 className="text-lg font-medium mb-2">Visual Style</h3>
              <p className="text-sm">{visualPlan.style.description}</p>
              
              {visualPlan.style.elements && (
                <div className="grid grid-cols-2 gap-3 mt-3">
                  {Object.entries(visualPlan.style.elements).map(([key, value]: [string, any]) => (
                    <div key={key} className="p-2 bg-muted/30 rounded">
                      <h4 className="text-xs font-medium capitalize">{key.replace('_', ' ')}</h4>
                      <p className="text-xs">{value}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
          
          {/* Scene Flow */}
          {visualPlan.scene_flow && (
            <div className="p-4 bg-muted/20 rounded-md">
              <h3 className="text-lg font-medium mb-2">Scene Flow</h3>
              <p className="text-sm">{visualPlan.scene_flow.principle}</p>
              
              {visualPlan.scene_flow.transitions && (
                <div className="mt-3">
                  <h4 className="text-sm font-medium mb-1">Transitions</h4>
                  <p className="text-xs">{visualPlan.scene_flow.transitions}</p>
                </div>
              )}
            </div>
          )}
          
          {/* Visual Hooks */}
          {visualPlan.visual_hooks && (
            <div className="p-4 bg-muted/20 rounded-md">
              <h3 className="text-lg font-medium mb-2">Visual Hooks</h3>
              <p className="text-sm">{visualPlan.visual_hooks.primary_hook}</p>
              
              {visualPlan.visual_hooks.elements && (
                <div className="mt-3">
                  <h4 className="text-sm font-medium mb-1">Elements</h4>
                  <ul className="list-disc list-inside space-y-1">
                    {visualPlan.visual_hooks.elements.map((element: string, index: number) => (
                      <li key={index} className="text-xs">{element}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </motion.div>
      )}
      </AnimatePresence>
    </motion.div>
  );
}