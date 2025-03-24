'use client';

import { useState, useId } from 'react';

interface PromptTemplateViewerProps {
  templates?: {
    script_generation?: string;
    hook_image?: string;
    content_analysis?: string;
    metadata_generation?: string;
    visual_planning?: string;
    image_prompt?: string;
    content_strategy?: string;
  };
}

export default function PromptTemplateViewer({ templates }: PromptTemplateViewerProps) {
  const [activeTemplate, setActiveTemplate] = useState<string | null>('script_generation');
  const selectId = useId();
  
  // Early return if no templates
  if (!templates || Object.keys(templates).length === 0) {
    return (
      <div className="p-4 bg-accent/40 rounded-md text-center">
        <p className="text-muted-foreground text-sm">No prompt templates available</p>
      </div>
    );
  }
  
  // Template names for display
  const templateNames: Record<string, string> = {
    script_generation: 'Script Generation',
    hook_image: 'Hook Image',
    content_analysis: 'Content Analysis',
    metadata_generation: 'Metadata',
    visual_planning: 'Visual Planning',
    image_prompt: 'Image Prompt',
    content_strategy: 'Content Strategy'
  };
  
  // Get available templates
  const availableTemplates = Object.keys(templates).filter(key => templates[key as keyof typeof templates]);
  
  // Get currently selected template content
  const selectedTemplate = activeTemplate ? templates[activeTemplate as keyof typeof templates] : null;
  
  return (
    <div className="bg-accent/30 rounded-lg overflow-hidden">
      {/* Template selector */}
      <div className="border-b border-border p-2">
        <label htmlFor={`template-select-${selectId}`} className="block text-xs font-medium mb-1">
          Select Template
        </label>
        <select
          id={`template-select-${selectId}`}
          value={activeTemplate || ''}
          onChange={(e) => setActiveTemplate(e.target.value)}
          className="w-full p-2 text-sm border border-input rounded-md bg-transparent"
        >
          {availableTemplates.map((key) => (
            <option key={key} value={key}>
              {templateNames[key] || key}
            </option>
          ))}
        </select>
      </div>
      
      {/* Template content */}
      <div className="p-4">
        {selectedTemplate ? (
          <div className="prose prose-sm max-w-none">
            <pre className="text-xs overflow-auto bg-accent p-3 rounded-md max-h-[400px]">
              {selectedTemplate}
            </pre>
          </div>
        ) : (
          <p className="text-muted-foreground text-sm">Select a template to view</p>
        )}
      </div>
    </div>
  );
}