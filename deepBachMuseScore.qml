import QtQuick 2.2
import QtQuick.Dialogs 1.2
import QtQuick.Controls 1.1
import MuseScore 1.0
import FileIO 1.0

MuseScore {
    id: mainMuseScoreObj
    menuPath: "Plugins.deepBachMuseScore"
    description: qsTr("This plugin calls deepBach project.")
    pluginType: "dock"
    dockArea:   "left"
    property variant serverCalled: false
    property variant loading: false
    property variant linesLogged: 0
    FileIO {
        id: myFile
        source: tempPath() + "/" + mainMuseScoreObj.curScore.title + "_recomposed_by_deepBach.xml"
        onError: console.log(msg)
    }
    onRun: {
        console.log('on run called')
        if (mainMuseScoreObj.serverCalled !== serverAddressInput.text || modelSelector.model.length === 0) {
            var requestModels = getRequestObj('GET', 'models')
            if (call(
                requestModels,
                false,
                function(responseText){
                    mainMuseScoreObj.serverCalled = serverAddressInput.text;
                    try {
                        modelSelector.model = JSON.parse(responseText)
                        logStatus('Models list loaded')
                        var requestLoadModel = getRequestObj("GET", 'current_model')
                        call(
                            requestLoadModel,
                            false,
                            function(response) {
                                logStatus('currently loaded model is ' + response)
                                for (var i in modelSelector.model) {
                                    if (modelSelector.model[i] === response) {
                                        console.log('set selected at ' + i)
                                        modelSelector.currentIndex = i
                                    }
                                }
                            }
                        )
                    } catch(error) {
                        console.log(error)
                        logStatus('No models found')
                    }

                }
            )) {
                logStatus('Retrieving models list at ' + serverAddressInput.text)
            }
        }
    }
    Rectangle {
        id: wrapperPanel
        color: "white"
        Text {
            id: title
            text: "Deep Bach"
            font.family: "Helvetica"
            font.pointSize: 24
            color: "green"
            anchors.top: wrapperPanel.top
            anchors.topMargin: 10
            font.underline: true
        }
        Label {
            id: serverInputLabel
            wrapMode: Text.WordWrap
            text: 'Server address'
            font.pointSize:12
            anchors.left: wrapperPanel.left
            anchors.top: title.top
            anchors.topMargin: 35
        }
        TextField {
            id: serverAddressInput
            text: "http://localhost:5000/"
            anchors.left: wrapperPanel.left
            anchors.top: serverInputLabel.top
            anchors.topMargin: 15
            width: 200
            height: 20
            onEditingFinished: {
                console.log('editing finished')
                mainMuseScoreObj.onRun();
            } 
        }
        ComboBox {
            id: modelSelector
            anchors.top: serverAddressInput.bottom 
            anchors.topMargin: 15
            model: []
            width: 200
        }
        Button {
            id : loadModel
            anchors.top: modelSelector.top
            anchors.left: modelSelector.right
            anchors.leftMargin: 10
            text: qsTr("Load")
            onClicked: {
                if (modelSelector.model[modelSelector.currentIndex]) {
                    var requestLoadModel = getRequestObj("POST", 'current_model')
                    if (call(
                        requestLoadModel,
                        {
                            model_name: modelSelector.model[modelSelector.currentIndex]
                        }, 
                        function(response) {
                            logStatus(response)
                        }
                    )) {
                        logStatus('Loading model ' + modelSelector.model[modelSelector.currentIndex])
                    }
                }
            }
        }
        Button {
            id : buttonOpenFile
            text: qsTr("Compose")
            anchors.left: wrapperPanel.left
            anchors.top: modelSelector.top
            anchors.topMargin: 30
            anchors.bottomMargin: 10
            onClicked: {
                var cursor = curScore.newCursor();
                cursor.rewind(1);
                var startStaff = cursor.staffIdx;
                var startTick = cursor.tick;
                cursor.rewind(2);
                var endStaff = cursor.staffIdx;
                var endTick = cursor.tick;
                var extension = 'xml'
                // var targetFile = tempPath() + "/my_file." + extension
                myFile.remove();
                var res = writeScore(mainMuseScoreObj.curScore, myFile.source, extension)
                var request = getRequestObj("POST", 'compose')
                if (call(
                    request,
                    {
                        start_staff: startStaff,
                        end_staff: endStaff,
                        start_tick: startTick,
                        end_tick: endTick,
                        xml_string: myFile.read()
                    },
                    function(response) {
                        if (response) {
                            logStatus('Done composing')
                            myFile.write(response)
                            readScore(myFile.source)
                        } else {
                            logStatus('Got Empty Response when composing')
                        }
                    }
                )) {
                    logStatus('Composing...')
                }
            }
        }
        Label {
            id: statusLabel
            wrapMode: Text.WordWrap
            text: ''
            color: 'green'
            font.pointSize:12
            anchors.left: wrapperPanel.left
            anchors.top: buttonOpenFile.top
            anchors.leftMargin: 10
            anchors.topMargin: 30
        }
    }
    function logStatus(text) {
        mainMuseScoreObj.linesLogged++;
        if (mainMuseScoreObj.linesLogged > 15) {
            // break the textblock into an array of lines
            var lines = statusLabel.text.split("\r\n");
            // remove one line, starting at the first position
            lines.splice(0,1);
            // join the array back into a single string
            statusLabel.text = lines.join("\r\n");
        }
        statusLabel.text += '- ' + text + "\r\n"
    }
    function getRequestObj(method, endpoint) {
        console.debug('calling endpoint ' + endpoint)
        var request = new XMLHttpRequest()
        endpoint = endpoint || ''
        request.open(method, serverAddressInput.text + endpoint, true)
        return request
    }
    function call(request, params, cb) {
        if (mainMuseScoreObj.loading) {
            logStatus('refusing to call server')
            return false
        }
        request.onreadystatechange = function() {
            if (request.readyState == XMLHttpRequest.DONE) {
                mainMuseScoreObj.loading = false;
                cb(request.responseText);
            }
        }
        if (params) {
            request.setRequestHeader("Content-Type", "application/x-www-form-urlencoded")
            var pairs = [];
            for (var prop in params) {
              if (params.hasOwnProperty(prop)) {
                var k = encodeURIComponent(prop),
                    v = encodeURIComponent(params[prop]);
                pairs.push( k + "=" + v);
              }
            }

            const content = pairs.join('&')
            console.debug('params ' + content)
            mainMuseScoreObj.loading = true;
            request.send(content)
        } else {
            mainMuseScoreObj.loading = true;
            request.send()
        }
        return true
    }
}
