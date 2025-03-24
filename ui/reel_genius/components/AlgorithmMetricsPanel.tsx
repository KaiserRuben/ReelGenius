'use client';

import { useState, useId } from 'react';

interface AlgorithmMetricsProps {
  data?: {
    algorithm_metrics?: {
      engagement_triggers?: string[];
      retention_metrics?: {
        exit_points_prevented?: number;
        psychological_anchors?: string[];
        open_loops?: string[];
        pattern_interrupts?: string[];
      };
      sharing_metrics?: {
        identity_triggers?: string[];
        social_currency?: string;
        estimated_share_rate?: number;
      };
      visual_optimization?: {
        algorithm_signals?: string[];
        attention_zones?: string[];
        color_psychology?: Record<string, string>;
      };
      hook_effectiveness?: {
        pattern_interrupt_strength?: number;
        cognitive_tension?: number;
        attention_capture_score?: number;
      };
    };
    prompt_templates?: {
      script_generation?: string;
      hook_image?: string;
      content_analysis?: string;
      metadata_generation?: string;
      visual_planning?: string;
    };
    metadata?: {
      hashtags?: string[];
      algorithmic_triggers?: string[];
      title?: string;
      description?: string;
    };
  };
}

export default function AlgorithmMetricsPanel({ data }: AlgorithmMetricsProps) {
  const [activeTab, setActiveTab] = useState<'engagement' | 'retention' | 'sharing' | 'visual'>('engagement');
  const tabId = useId();
  
  // Early return if no algorithm data
  if (!data?.algorithm_metrics) {
    return (
      <div className="p-4 bg-accent/40 rounded-md text-center">
        <p className="text-muted-foreground text-sm">No algorithm optimization data available</p>
      </div>
    );
  }
  
  const metrics = data.algorithm_metrics;
  
  return (
    <div className="bg-accent/30 rounded-lg overflow-hidden">
      {/* Tabs */}
      <div className="flex border-b border-border">
        <button
          id={`${tabId}-engagement`}
          onClick={() => setActiveTab('engagement')}
          className={`px-4 py-2 text-sm font-medium ${activeTab === 'engagement' ? 'bg-primary/10 text-primary border-b-2 border-primary' : 'text-muted-foreground'}`}
        >
          Engagement
        </button>
        <button
          id={`${tabId}-retention`}
          onClick={() => setActiveTab('retention')}
          className={`px-4 py-2 text-sm font-medium ${activeTab === 'retention' ? 'bg-primary/10 text-primary border-b-2 border-primary' : 'text-muted-foreground'}`}
        >
          Retention
        </button>
        <button
          id={`${tabId}-sharing`}
          onClick={() => setActiveTab('sharing')}
          className={`px-4 py-2 text-sm font-medium ${activeTab === 'sharing' ? 'bg-primary/10 text-primary border-b-2 border-primary' : 'text-muted-foreground'}`}
        >
          Sharing
        </button>
        <button
          id={`${tabId}-visual`}
          onClick={() => setActiveTab('visual')}
          className={`px-4 py-2 text-sm font-medium ${activeTab === 'visual' ? 'bg-primary/10 text-primary border-b-2 border-primary' : 'text-muted-foreground'}`}
        >
          Visual
        </button>
      </div>
      
      {/* Tab Content */}
      <div className="p-4">
        {activeTab === 'engagement' && (
          <div className="space-y-4">
            <div>
              <h3 className="font-medium text-sm mb-2">Engagement Triggers</h3>
              <div className="flex flex-wrap gap-1.5">
                {metrics.engagement_triggers?.map((trigger, index) => (
                  <span key={index} className="px-2 py-1 bg-primary/10 text-primary text-xs rounded-full">
                    {trigger}
                  </span>
                )) || <span className="text-xs text-muted-foreground">No engagement triggers data</span>}
              </div>
            </div>
            
            {metrics.hook_effectiveness && (
              <div>
                <h3 className="font-medium text-sm mb-2">Hook Effectiveness</h3>
                <div className="grid grid-cols-3 gap-2">
                  <div className="bg-accent/50 p-2 rounded-md text-center">
                    <div className="text-2xl font-bold">
                      {Math.round(metrics.hook_effectiveness.pattern_interrupt_strength * 100) || 0}%
                    </div>
                    <div className="text-xs text-muted-foreground">Pattern Interrupt</div>
                  </div>
                  <div className="bg-accent/50 p-2 rounded-md text-center">
                    <div className="text-2xl font-bold">
                      {Math.round(metrics.hook_effectiveness.cognitive_tension * 100) || 0}%
                    </div>
                    <div className="text-xs text-muted-foreground">Cognitive Tension</div>
                  </div>
                  <div className="bg-accent/50 p-2 rounded-md text-center">
                    <div className="text-2xl font-bold">
                      {Math.round(metrics.hook_effectiveness.attention_capture_score * 100) || 0}%
                    </div>
                    <div className="text-xs text-muted-foreground">Attention Capture</div>
                  </div>
                </div>
              </div>
            )}
            
            {data.metadata?.hashtags && (
              <div>
                <h3 className="font-medium text-sm mb-2">Algorithmic Hashtags</h3>
                <div className="flex flex-wrap gap-1.5">
                  {data.metadata.hashtags.map((tag, index) => (
                    <span key={index} className="px-2 py-1 bg-blue-500/10 text-blue-500 text-xs rounded-full">
                      #{tag}
                    </span>
                  ))}
                </div>
              </div>
            )}
            
            {data.metadata?.algorithmic_triggers && (
              <div>
                <h3 className="font-medium text-sm mb-2">Algorithmic Triggers</h3>
                <div className="flex flex-wrap gap-1.5">
                  {data.metadata.algorithmic_triggers.map((trigger, index) => (
                    <span key={index} className="px-2 py-1 bg-purple-500/10 text-purple-500 text-xs rounded-full">
                      {trigger}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
        
        {activeTab === 'retention' && (
          <div className="space-y-4">
            {metrics.retention_metrics?.exit_points_prevented !== undefined && (
              <div>
                <h3 className="font-medium text-sm mb-2">Exit Points Prevented</h3>
                <div className="bg-green-500/10 text-green-500 p-3 rounded-md text-center">
                  <span className="text-2xl font-bold">{metrics.retention_metrics.exit_points_prevented}</span>
                </div>
              </div>
            )}
            
            {metrics.retention_metrics?.psychological_anchors && (
              <div>
                <h3 className="font-medium text-sm mb-2">Psychological Anchors</h3>
                <div className="grid grid-cols-1 gap-2">
                  {metrics.retention_metrics.psychological_anchors.map((anchor, index) => (
                    <div key={index} className="bg-accent/50 p-2 rounded-md text-sm">
                      {anchor}
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {metrics.retention_metrics?.open_loops && (
              <div>
                <h3 className="font-medium text-sm mb-2">Open Loops</h3>
                <div className="grid grid-cols-1 gap-2">
                  {metrics.retention_metrics.open_loops.map((loop, index) => (
                    <div key={index} className="bg-accent/50 p-2 rounded-md text-sm">
                      {loop}
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {metrics.retention_metrics?.pattern_interrupts && (
              <div>
                <h3 className="font-medium text-sm mb-2">Pattern Interrupts</h3>
                <div className="grid grid-cols-1 gap-2">
                  {metrics.retention_metrics.pattern_interrupts.map((interrupt, index) => (
                    <div key={index} className="bg-accent/50 p-2 rounded-md text-sm">
                      {interrupt}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
        
        {activeTab === 'sharing' && (
          <div className="space-y-4">
            {metrics.sharing_metrics?.estimated_share_rate !== undefined && (
              <div>
                <h3 className="font-medium text-sm mb-2">Estimated Share Rate</h3>
                <div className="bg-blue-500/10 text-blue-500 p-3 rounded-md text-center">
                  <span className="text-2xl font-bold">
                    {(metrics.sharing_metrics.estimated_share_rate * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            )}
            
            {metrics.sharing_metrics?.identity_triggers && (
              <div>
                <h3 className="font-medium text-sm mb-2">Identity Triggers</h3>
                <div className="flex flex-wrap gap-1.5">
                  {metrics.sharing_metrics.identity_triggers.map((trigger, index) => (
                    <span key={index} className="px-2 py-1 bg-primary/10 text-primary text-xs rounded-full">
                      {trigger}
                    </span>
                  ))}
                </div>
              </div>
            )}
            
            {metrics.sharing_metrics?.social_currency && (
              <div>
                <h3 className="font-medium text-sm mb-2">Social Currency</h3>
                <div className="bg-accent/50 p-3 rounded-md text-sm">
                  {metrics.sharing_metrics.social_currency}
                </div>
              </div>
            )}
            
            {data.metadata?.title && (
              <div>
                <h3 className="font-medium text-sm mb-2">Optimized Title</h3>
                <div className="bg-accent/50 p-3 rounded-md font-medium">
                  {data.metadata.title}
                </div>
              </div>
            )}
          </div>
        )}
        
        {activeTab === 'visual' && (
          <div className="space-y-4">
            {metrics.visual_optimization?.algorithm_signals && (
              <div>
                <h3 className="font-medium text-sm mb-2">Algorithm Signals</h3>
                <div className="flex flex-wrap gap-1.5">
                  {metrics.visual_optimization.algorithm_signals.map((signal, index) => (
                    <span key={index} className="px-2 py-1 bg-primary/10 text-primary text-xs rounded-full">
                      {signal}
                    </span>
                  ))}
                </div>
              </div>
            )}
            
            {metrics.visual_optimization?.attention_zones && (
              <div>
                <h3 className="font-medium text-sm mb-2">Attention Zones</h3>
                <div className="grid grid-cols-1 gap-2">
                  {metrics.visual_optimization.attention_zones.map((zone, index) => (
                    <div key={index} className="bg-accent/50 p-2 rounded-md text-sm">
                      {zone}
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {metrics.visual_optimization?.color_psychology && (
              <div>
                <h3 className="font-medium text-sm mb-2">Color Psychology</h3>
                <div className="grid grid-cols-1 gap-2">
                  {Object.entries(metrics.visual_optimization.color_psychology).map(([color, meaning], index) => (
                    <div key={index} className="flex items-center gap-2 bg-accent/50 p-2 rounded-md">
                      <div 
                        className="w-4 h-4 rounded-full" 
                        style={{ backgroundColor: color.startsWith('#') ? color : `#${color}` }}
                      />
                      <span className="text-sm">{meaning}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}