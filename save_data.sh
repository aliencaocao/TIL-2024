# read -p "Enter the id of the docker container: " dockerCont
read -p "Enter the name of the output directory: " output

command_output=$(sudo docker ps)
echo "part 1"
while IFS= read -r line; do
    
    if [[ $line == *"til_competition"* ]]; then

        echo $line
        dockerCont=$(echo $line | head -n1 | awk '{print $1}')
        echo "Line containing 'til_competition': $line"
        echo "First part of the line: $dockerCont"
        echo "Using Docker ID: $dockerCont"

        echo "Creating $output"
        mkdir "localTestResults/$output"

        docker cp $dockerCont:results/team_12000sgdplushie_results.jsonl localTestResults/$output/results.jsonl

        for i in $(seq 0 4);
        do
            docker cp $dockerCont:results/team_12000sgdplushie_snapshot_$i.jpg localTestResults/$output/$i.jpg
        done

    fi
done <<< "$command_output"

docker compose down
