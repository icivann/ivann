<template>
  <div class="box d-sm-flex">
    {{value}}
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

@Component
export default class IntegerInc extends Vue {
  @Prop() value!: number;

  @Prop() index!: number;

  increment() {
    this.$emit('value-change', this.value + 1, this.index);
  }

  decrement() {
    this.$emit('value-change', this.value - 1, this.index);
  }
}
</script>

<style scoped>
  .box {
    margin: 0 5px;
    padding: 0 0 0 0.3em;
    background: #ececec;
    border-radius: 2px;
    color: #303030;
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

  .inc-button:hover {
    background: #e0e0e0e0;
  }
</style>
