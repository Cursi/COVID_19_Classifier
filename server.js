const express = require('express')
const app = express()
const port = process.env.PORT || 3000

app.get('/', (req, res) => 
{
    const { spawn } = require('child_process');
    // const pyProg = spawn('python', ['./dummy.py']);
    const pyProg = spawn('heroku', ['run', 'python', './dummy.py']);

    pyProg.stdout.on('data', function(data) 
    {
        console.log(data.toString());
        res.write(data);
        res.end();
    });
})

app.listen(port, () => 
{
    console.log(`Example app listening on ${port}`)
})