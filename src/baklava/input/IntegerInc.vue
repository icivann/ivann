<template>
  <div class="box" ref="box" tabindex="1">
    <div class="text-display" @click="editOn" v-if="!edit" ref="text">
      {{value}}
    </div>
    <input @focus="$event.target.select()" v-model="editValue" v-show="edit" tabindex="0"
           v-on:keyup.enter="enter" @focusout="focusOut" ref="input"
           :style="'width: ' + inputBoxWidth + 'px'">
    <div class="buttons">
      <div class="inc-button" @click="increment">
        <svg xmlns="http://www.w3.org/2000/svg" width="6" height="4"
             viewBox="0 0 7 4">
          <line x1="0.5" y1="4" x2="3.5" y2="1" fill="none" stroke="#202020" stroke-width="1"/>
          <line x1="6" x2="3" y1="4" y2="1" fill="none" stroke="#202020" stroke-width="1"/>
        </svg>
      </div>
      <div class="inc-button" @click="decrement">
        <svg xmlns="http://www.w3.org/2000/svg" width="6" height="4"
             viewBox="0 0 7 4">
          <line x1="0.5" y2="3" x2="3.5" fill="none" stroke="#202020" stroke-width="1"/>
          <line x1="6" x2="3" y2="3" fill="none" stroke="#202020" stroke-width="1"/>
        </svg>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import Focusable from '@/baklava/Focusable';

@Component
export default class IntegerInc extends Vue {
  @Prop() value!: number;

  @Prop() index!: number;

  private edit = false;
  private editValue = '';
  private inputBoxWidth = 0;

  private increment() {
    this.$emit('value-change', this.value + 1, this.index);
  }

  private decrement() {
    this.$emit('value-change', this.value - 1, this.index);
  }

  private updateValue() {
    this.$emit('value-change', parseInt(this.editValue, 10), this.index);
  }

  private focusOut() {
    this.updateValue();
    this.toggle();
  }

  private enter() {
    (this.$refs.box as Focusable).focus();
    this.updateValue();
  }

  private editOn() {
    this.editValue = this.value.toString();
    this.inputBoxWidth = (this.$refs.text as Vue & { clientWidth: number }).clientWidth;
    this.toggle();
    this.$nextTick(() => {
      (this.$refs.input as Focusable).focus();
    });
  }

  private toggle() {
    this.edit = !this.edit;
  }
}
</script>

<style scoped>
  .box {
    margin: 0 3px 0 5px;
    padding: 0 0 0 0.3em;
    background: #ececec;
    border-radius: 2px;
    color: #303030;
    display: flex;
  }

  .buttons {
    font-size: 0.5em;
    text-align: center;
    margin-left: 0.3em;
    padding-bottom: 0.2em;
  }

  .inc-button {
    border-radius: 2px;
    padding: 0 0.3em 0 0.1em;
  }

  .inc-button:hover, .text-display:hover, input:hover {
    background: #e0e0e0e0;
  }

  input {
    width: fit-content;
    padding: 0;
    margin: 0;
    border-style: none;
    background: #ececec;
  }

  textarea:focus, input:focus, .box:focus {
    outline: none;
  }

  .text-display {
    padding-right: 0.5em;
  }

  input::selection {
    background: #303030;
    color: #e0e0e0;
  }
</style>
