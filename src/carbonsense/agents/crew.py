def _validate_llm_params(self, params: dict, agent_name: str) -> dict:
        """Validate LLM parameters from YAML configuration."""
        validated = params.copy()
        
        # Validate numerical ranges
        if 'temperature' in validated:
            if not 0 <= validated['temperature'] <= 1:
                raise ValueError(f"Temperature must be between 0 and 1 for agent {agent_name}")
        
        if 'presence_penalty' in validated:
            if not -2 <= validated['presence_penalty'] <= 2:
                raise ValueError(f"Presence penalty must be between -2 and 2 for agent {agent_name}")
        
        if 'frequency_penalty' in validated:
            if not -2 <= validated['frequency_penalty'] <= 2:
                raise ValueError(f"Frequency penalty must be between -2 and 2 for agent {agent_name}")
        
        if 'top_p' in validated:
            if not 0 <= validated['top_p'] <= 1:
                raise ValueError(f"Top_p must be between 0 and 1 for agent {agent_name}")
        
        # Validate types
        if 'n' in validated:
            validated['n'] = int(validated['n'])
            if validated['n'] < 1:
                raise ValueError(f"N must be positive integer for agent {agent_name}")
        
        if 'max_tokens' in validated:
            validated['max_tokens'] = int(validated['max_tokens'])
            if validated['max_tokens'] < 1:
                raise ValueError(f"Max tokens must be positive integer for agent {agent_name}")
        
        if 'reasoning_effort' in validated:
            valid_efforts = ['none', 'low', 'medium', 'high']
            if validated['reasoning_effort'] not in valid_efforts:
                raise ValueError(f"Reasoning effort must be one of {valid_efforts} for agent {agent_name}")
        
        if 'timeout' in validated:
            validated['timeout'] = float(validated['timeout'])
            if validated['timeout'] <= 0:
                raise ValueError(f"Timeout must be positive for agent {agent_name}")
        
        if 'stop' in validated and not isinstance(validated['stop'], list):
            if isinstance(validated['stop'], str):
                validated['stop'] = [validated['stop']]
            else:
                raise ValueError(f"Stop sequences must be a string or list of strings for agent {agent_name}")
        
        if 'logit_bias' in validated and not isinstance(validated['logit_bias'], dict):
            raise ValueError(f"Logit bias must be a dictionary for agent {agent_name}")
        
        return validated

# Initialize LLM configurations for each agent from ai_parameters.yaml
        self.agent_llms = {}
        default_config = self.ai_params.get('default', {})
        
        for agent_name, agent_config in self.ai_params['agents'].items():
            # Merge default config with agent-specific config
            llm_config = default_config.copy()
            llm_config.update(agent_config)
            
            # Extract all possible LLM parameters
            llm_params = {
                # Core model parameters
                'model': llm_config.get('model'),
                'temperature': llm_config.get('temperature'),
                'top_p': llm_config.get('top_p'),
                'n': llm_config.get('n'),
                'max_tokens': llm_config.get('max_tokens'),
                'presence_penalty': llm_config.get('presence_penalty'),
                'frequency_penalty': llm_config.get('frequency_penalty'),
                'logit_bias': llm_config.get('logit_bias'),
                'stop': llm_config.get('stop'),
                'stream': llm_config.get('stream'),
                'timeout': llm_config.get('timeout'),
                
                # Advanced parameters
                'reasoning_effort': llm_config.get('reasoning_effort'),
                'seed': llm_config.get('seed'),
                'logprobs': llm_config.get('logprobs'),
                'top_logprobs': llm_config.get('top_logprobs'),
                
                # API configuration from watsonx_config
                'api_base': watsonx_config.get('api_base'),
                'api_key': watsonx_config.get('api_key'),
                'api_version': watsonx_config.get('api_version'),
                'base_url': watsonx_config.get('base_url'),
            }
            
            # Remove None values to use class defaults
            llm_params = {k: v for k, v in llm_params.items() if v is not None}
            
            # Validate parameters
            llm_params = self._validate_llm_params(llm_params, agent_name)
            
            # Create LLM instance with all supported parameters
            self.agent_llms[agent_name] = LLM(**llm_params)