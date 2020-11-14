const express = require('express')
const app = express()
const port = process.env.PORT || 3000

// var cmd=require('node-cmd');
const PS = require('python-shell');

app.get('/', (req, res) => 
{
    // const { spawn } = require('child_process');
    // // const pyProg = spawn('python', ['./dummy.py']);
    // const pyProg = spawn('heroku', ['run', 'python', './dummy.py']);

    // const pyProg = cmd.run("python dummy.py")

    PS.PythonShell.run('dummy.py', null, function (err) 
    {
        if (err) throw err;
            console.log('finished');
    }).stdout.on('data', function(data) 
    {
        console.log(data.toString());
        res.write(data);
        res.end();
    });

    // pyProg.stdout.on('data', function(data) 
    // {
    //     console.log(data.toString());
    //     res.write(data);
    //     res.end();
    // });
})

app.listen(port, () => 
{
    console.log(`Example app listening on ${port}`)
})