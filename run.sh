python3 src/linkedin_analyzer/linkedin_analyzer.py \
    analyze \
    --config config/config.yml \
    --connections input/Connections.csv \
    --messages input/Messages.csv \
    --profile input/Profile.csv \
    --output linkedin_insights \
    --output-dir output/ \
    --format html
