import json
import os

target = r'C:\Users\Sukku\.gemini\antigravity\brain\f45f950b-c7a9-48b9-a717-d50a9084b029\.system_generated\logs\transcript_full.jsonl'
files_to_restore = {
    r'c:\credit\predict.py': 'predict.py.bak',
    r'c:\credit\templates\predict.html': 'predict.html.bak',
    r'c:\credit\templates\result.html': 'result.html.bak'
}

with open(target, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            if 'tool_calls' in data:
                # We only want to restore from before 08:41:00Z
                if data.get('created_at') > "2026-07-01T08:41:00Z":
                    continue
                    
                for call in data['tool_calls']:
                    if call['name'] == 'write_to_file':
                        args = call.get('args', {})
                        file_path = args.get('TargetFile', '').lower()
                        if file_path in files_to_restore:
                            with open(files_to_restore[file_path], 'w', encoding='utf-8') as out:
                                out.write(args.get('CodeContent', ''))
        except Exception as e:
            pass
