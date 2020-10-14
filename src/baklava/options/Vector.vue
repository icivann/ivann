<template>
  <div class="d-sm-flex">
    <div class="ml-1">{{ name }}</div>
    <IntegerInc :index=index
                :key="index"
                :value=val
                @increment="increment"
                v-for="(val, index) in value"></IntegerInc>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import IntegerInc from '@/baklava/input/IntegerInc.vue';

@Component({
  components: { IntegerInc },
})
export default class Vector extends Vue {
  @Prop() value!: [number];

  @Prop({ type: String }) name!: string;

  increment(x: number) {
  // increment(x: number, index: number) {
    const copy = [...this.value];
    copy[0] = x;

    this.$emit('input', copy);
    console.log(x);
  }
}
</script>
