import { Elemental, passAlongProps } from "https://esm.sh/gh/jeff-hykin/elemental@0.6.5/main/deno.js"
import { css, components, Column, Row, askForFiles, Code, Input, Checkbox, Dropdown, popUp, cx, } from "https://esm.sh/gh/jeff-hykin/good-component@0.3.0/elements.js"
import { fadeIn, fadeOut } from "https://esm.sh/gh/jeff-hykin/good-component@0.3.0/main/animations.js"
import { showToast } from "https://esm.sh/gh/jeff-hykin/good-component@0.3.0/main/actions.js"
import { addDynamicStyleFlags, setupStyles, createCssClass, setupClassStyles, hoverStyleHelper, combineClasses, mergeStyles, AfterSilent, removeAllChildElements } from "https://esm.sh/gh/jeff-hykin/good-component@0.3.0/main/helpers.js"
import { zip, enumerate, count, permute, combinations, wrapAroundGet } from "https://esm.sh/gh/jeff-hykin/good-js@1.13.5.1/source/array.js"

import storageObject from "https://esm.sh/gh/jeff-hykin/storage-object@0.0.3.5/main.js"
import { makeDraggable } from "./generic_tools.js"

import { Event, trigger, everyTime, once } from "https://esm.sh/gh/jeff-hykin/good-js@1.14.3.3/source/events.js"
import { pointsToFunction } from "https://esm.sh/gh/jeff-hykin/good-js@1.14.3.3/source/flattened/points_to_function.js"
import { fabric, FabricCanvas } from "./fabric.js"
import { NodeCanvas, FabricNode } from "./node_canvas.js"
import { TimelineManager } from "./timeline_manager.js"
import { Button } from "./button.js"
globalThis.fabric = fabric // debugging

// goals
    // pulse animation
    // node visual connections
    // infinite pan and zoom canvas

// 
// node
// 
    const hslBlueHue = 213
    const hslYellowHue = 49
    const hslRedHue = 14
    const energyToHue = pointsToFunction({
        xValues: [0, 1],
        yValues: [hslBlueHue, hslYellowHue],
        areSorted: true,
        method: "linearInterpolation",
    })
    const pulseAnimation = [
        [
            { transform: 'scale(1)', opacity: 1 },  // Initial state
            { transform: 'scale(1.2)', opacity: 0.8 }, // Pulsed state (enlarged and slightly faded)
            { transform: 'scale(1)', opacity: 1 },  // Back to original state
        ],
        {
            duration: 600,
            iterations: 1,
            easing: 'ease-in-out',
        }
    ]
    function Node({ label, x, y, threshold=1, startingEnergy=0, energyDecayRate=0.1, onPositionChange, ...props }) {
        const element = document.createElement("div")
        Object.assign(element.style, {
            backgroundColor: `hsl(${hslBlueHue}, 100%, 50%)`,
            color: "white",
            position: "absolute",
            left: `${x}px`,
            top: `${y}px`,
            width: "100px",
            height: "100px",
            borderRadius: "5rem",
            border: "5px solid lightgray",
            cursor: "pointer",
            transition: "background-color 0.2s ease",
        })
        Object.assign(element, {
            outputEvent: new Event(),
            _inputEvents: new Set(),
            energy: startingEnergy,
            threshold,
            energyDecayRate,
            _updateColorBasedOnEnergy: (energy)=>{
                const relativeEnergy = energy / element.threshold
                if (energy >= 1) {
                    element.style.backgroundColor = `hsl(${hslRedHue}, 100%, 50%)`
                    let animation = element.animate(
                        ...pulseAnimation
                    )
                    animation.onfinish = ()=>{
                        // reset to min color
                        element._updateColorBasedOnEnergy(Node.energyAfterFiring)
                    }
                } else {
                    element.style.backgroundColor = `hsl(${energyToHue(relativeEnergy)}, 100%, 50%)`
                }
            },
            _updateTickCallback: ()=>{
                console.debug(`element.energy is:`,element.energy)
                if (element.energy > element.threshold) {
                    trigger(element.outputEvent)
                    element._updateColorBasedOnEnergy(element.threshold+1) // show red and pulse
                    element.energy = Node.energyAfterFiring
                } else {
                    element.energy -= element.energyDecayRate
                    if (element.energy < Node.stableEnergyLevel) {
                        element.energy = Node.stableEnergyLevel
                    }
                }
            },
            _inputResponse: (input)=>{
                element.energy += input.intensity
                element._updateColorBasedOnEnergy(element.energy)
            },
            addInput: (input, intensity) => {
                everyTime(input).then(element._inputResponse)
                element._inputEvents.add(input)
            },
            removeInput: (input) => {
                // disconnect hook
                input.remove(element._inputResponse)
                // remove from set
                element._inputEvents.delete(input)
            },
        })
        makeDraggable(
            element,
            {
                onDrag: ({isStart, isEnd, x, y})=>{
                    if (onPositionChange) {
                        onPositionChange({isStart, isEnd, x, y})
                    }
                }, 
                itsPositionedCorrectlyIPromise: true,
            }
        )
        Node.allNodes.add(new WeakRef(element))
        return passAlongProps(element, props)
    }
    globalThis.Node = Node
    Object.assign(Node, {
        allNodes: new Set(),
        updateInterval: 1000,
        energyAfterFiring: 0,
        stableEnergyLevel: 0.1,
    })
    
    const canvasElement = NodeCanvas({ backgroundColor: "whitesmoke", jsonObjects: {
            // FIXME: debugging
            "objects": [
                // {
                //     "type": "rect",
                //     "left": 50,
                //     "top": 50,
                //     "width": 20,
                //     "height": 20,
                //     "fill": "green",
                //     "overlayFill": null,
                //     "stroke": null,
                //     "strokeWidth": 1,
                //     "strokeDashArray": null,
                //     "scaleX": 1,
                //     "scaleY": 1,
                //     "angle": 0,
                //     "flipX": false,
                //     "flipY": false,
                //     "opacity": 1,
                //     "selectable": true,
                //     "hasControls": true,
                //     "hasBorders": true,
                //     "hasRotatingPoint": false,
                //     "transparentCorners": true,
                //     "perPixelTargetFind": false,
                //     "rx": 0,
                //     "ry": 0
                // },
                // {
                //     "type": "circle",
                //     "left": 100,
                //     "top": 100,
                //     "width": 100,
                //     "height": 100,
                //     "fill": "red",
                //     "overlayFill": null,
                //     "stroke": "rgba(100,200,200)",
                //     "strokeWidth": 5,
                //     "strokeDashArray": null,
                //     "scaleX": 1,
                //     "scaleY": 1,
                //     "angle": 0,
                //     "flipX": false,
                //     "flipY": false,
                //     "opacity": 1,
                //     "selectable": true,
                //     "hasControls": true,
                //     "hasBorders": true,
                //     "hasRotatingPoint": false,
                //     "transparentCorners": true,
                //     "perPixelTargetFind": false,
                //     "radius": 50
                // },
                new FabricNode({
                    id: "node-1",
                    label: "A", 
                    left:300, 
                    top:300, 
                    radius:50, 
                    spikeThreshold:1,
                    startingEnergy:0, 
                    energyDecayRate:0.1,
                }),
                new FabricNode({
                    id: "node-2",
                    label: "A", 
                    left:500, 
                    top:300, 
                    radius:50, 
                    spikeThreshold:1,
                    startingEnergy:0, 
                    energyDecayRate:0.1,
                    outputNodeIds: [
                        "node-1" 
                    ],
                }),
            ],
            // "background": "rgba(0, 0, 0, 0)",
        }
    })
    globalThis.canvasElement = canvasElement // debugging
    
    // 
    // Main node updater
    // 
    setInterval(() => {
        for (let eachRef of Node.allNodes) {
            let each = eachRef.deref()
            // remove garbage collected nodes (if we didnt use WeakRefs, this would be a memory leak)
            if (!each) {
                Node.allNodes.delete(eachRef)
            }
            // have nodes update visuals in unison
            each._updateTickCallback()
        }
    }, Node.updateInterval)


const { html } = Elemental({
    ...components,
    Node,
    FabricCanvas,
    Column, Row, askForFiles, Code, Input, Button, Checkbox, Dropdown,
})

document.body = html`
    <body font-size=15px background-color=whitesmoke overflow=none width=100vw height=100vh overflow=hidden>
        ${canvasElement}
        <Button
            style=${{position: "fixed", right: "10px", bottom: "10px"}}
            onclick=${async ()=>{
                console.log(`canvasElement is:`,canvasElement)
                if (!canvasElement.timelineManager) {
                    showToast("Sorry the canvas is not yet loaded", {close: true})
                    return
                }
                try {
                    await canvasElement.timelineManager.goForward()
                } catch (error) {
                    console.error(`error during goForward:`, error)
                }
            }}
            >
                Next
        </Button>
    </body>
`
        // <Node label="A" x=100 y=100></Node>